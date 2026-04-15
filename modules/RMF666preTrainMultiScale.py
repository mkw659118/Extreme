from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding


class StudentTMixturePrior(nn.Module):
    def __init__(
        self,
        num_components: int,
        state_dim: int,
        use_all_channels: bool = False,
        include_last_value: bool = True,
        scales: Sequence[int] = (1, 4, 8, 16),
        include_seq_level: bool = True,
        learnable_scale_weights: bool = True,
        min_scale: float = 1e-4,
        min_df: float = 2.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.state_dim = state_dim
        self.use_all_channels = use_all_channels
        self.include_last_value = include_last_value
        self.scales = tuple(scales)
        self.include_seq_level = include_seq_level
        self.min_scale = min_scale
        self.min_df = min_df
        self.temperature = temperature

        self.mu = nn.Parameter(torch.zeros(num_components, state_dim))
        self.scale_raw = nn.Parameter(torch.zeros(num_components, state_dim))
        self.df_raw = nn.Parameter(torch.zeros(num_components, state_dim))
        self.mix_logits = nn.Parameter(torch.zeros(num_components))
        self.register_buffer('_log_pi', torch.log(torch.tensor(torch.pi, dtype=torch.float32)), persistent=False)

        num_scale_terms = len(self.scales) + (1 if self.include_seq_level else 0)
        alpha_init = torch.zeros(num_scale_terms, dtype=torch.float32)
        if learnable_scale_weights:
            self.alpha_logits = nn.Parameter(alpha_init)
        else:
            self.register_buffer('alpha_logits', alpha_init, persistent=True)

    def extract_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        x_used = x if self.use_all_channels else x[:, :, :1]
        std = x_used.std(dim=1, unbiased=False).mean(dim=-1, keepdim=True)

        if x_used.size(1) > 1:
            dx = x_used[:, 1:, :] - x_used[:, :-1, :]
        else:
            dx = torch.zeros_like(x_used)

        max_abs_dx = dx.abs().amax(dim=1).mean(dim=-1, keepdim=True)
        mean_abs_dx = dx.abs().mean(dim=1).mean(dim=-1, keepdim=True)

        if self.include_last_value:
            last = x_used[:, -1, :].mean(dim=-1, keepdim=True)
            z = torch.cat([std, max_abs_dx, mean_abs_dx, last], dim=-1)
        else:
            z = torch.cat([std, max_abs_dx, mean_abs_dx], dim=-1)
        return z

    @staticmethod
    def _window_to_patches(x_used: torch.Tensor, patch_len: int) -> torch.Tensor:
        # x_used: [B, L, C] -> [B, Npatch, C], patch by non-overlap mean pooling
        batch_size, length, channels = x_used.shape
        if patch_len <= 1:
            return x_used

        usable = (length // patch_len) * patch_len
        if usable == 0:
            return x_used.mean(dim=1, keepdim=True)

        x_trim = x_used[:, :usable, :].contiguous()
        x_patch = x_trim.view(batch_size, usable // patch_len, patch_len, channels).mean(dim=2)
        return x_patch

    def _component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D], output: [B, K]
        mu = self.mu.unsqueeze(0)
        scale = F.softplus(self.scale_raw).unsqueeze(0) + self.min_scale
        df = F.softplus(self.df_raw).unsqueeze(0) + self.min_df
        log_pi = self._log_pi.to(device=df.device, dtype=df.dtype)

        z_expand = z.unsqueeze(1)
        log_norm = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * (torch.log(df) + log_pi)
            - torch.log(scale)
        )
        log_kernel = -((df + 1.0) / 2.0) * torch.log1p(((z_expand - mu) / scale).pow(2) / df)
        return (log_norm + log_kernel).sum(dim=-1)

    def posterior_from_z(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_comp = self._component_log_prob(z)
        log_pi = torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
        log_joint = log_comp + log_pi
        log_mix = torch.logsumexp(log_joint, dim=-1)
        q = torch.softmax(log_joint / self.temperature, dim=-1)
        return {
            'log_component': log_comp,
            'log_joint': log_joint,
            'log_mix': log_mix,
            'q': q,
            'mix_prob': torch.softmax(self.mix_logits, dim=0),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_used = x if self.use_all_channels else x[:, :, :1]

        z_scales = []
        log_comp_scales = []
        q_scales = []

        for patch_len in self.scales:
            x_scale = self._window_to_patches(x_used, patch_len=patch_len)
            z_scale = self.extract_state_vector(x_scale)
            z_scales.append(z_scale)

            log_comp_scale = self._component_log_prob(z_scale)
            log_comp_scales.append(log_comp_scale)

            q_scale = torch.softmax((log_comp_scale + torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)) / self.temperature, dim=-1)
            q_scales.append(q_scale)

        if self.include_seq_level:
            x_seq = x_used.mean(dim=1, keepdim=True)
            z_seq = self.extract_state_vector(x_seq)
            z_scales.append(z_seq)

            log_comp_seq = self._component_log_prob(z_seq)
            log_comp_scales.append(log_comp_seq)

            q_seq = torch.softmax((log_comp_seq + torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)) / self.temperature, dim=-1)
            q_scales.append(q_seq)

        alpha = torch.softmax(self.alpha_logits, dim=0)
        log_comp_stack = torch.stack(log_comp_scales, dim=1)  # [B, S, K]
        fused_log_comp = torch.sum(log_comp_stack * alpha.view(1, -1, 1), dim=1)  # [B, K]

        log_pi = torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
        log_joint = fused_log_comp + log_pi
        log_mix = torch.logsumexp(log_joint, dim=-1)
        q = torch.softmax(log_joint / self.temperature, dim=-1)

        out = {
            'log_component': fused_log_comp,
            'log_joint': log_joint,
            'log_mix': log_mix,
            'q': q,
            'mix_prob': torch.softmax(self.mix_logits, dim=0),
            'z': torch.stack(z_scales, dim=1).mean(dim=1),
            'z_scales': torch.stack(z_scales, dim=1),
            'q_scales': torch.stack(q_scales, dim=1),
            'alpha': alpha,
        }
        out['pretrain_nll'] = -log_mix.mean()
        return out


class RouterFromEmbeddingPreTrain(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = num_experts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, q_prior: torch.Tensor) -> torch.Tensor:
        return self.net(q_prior)


class LSTMExpert(nn.Module):
    def __init__(self, d_model: int, expert_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=max(1, expert_layers),
            batch_first=True,
            dropout=dropout if expert_layers > 1 else 0.0,
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out, _ = self.lstm(x)
        return self.norm(out + residual)


class BackboneMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        pred_len: int,
        out_dim: int,
        num_experts: int = 3,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_layers: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.pred_len = pred_len
        self.out_dim = out_dim

        self.experts = nn.ModuleList([
            LSTMExpert(d_model=d_model, expert_layers=expert_layers, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.fuse_norm = nn.RMSNorm(d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * out_dim),
        )

    def _build_sparse_topk_weights(self, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> torch.Tensor:
        batch_size, _ = head_mix_weights.shape
        full_weights = head_mix_weights.new_zeros((batch_size, self.num_experts))
        full_weights.scatter_(dim=1, index=topk_experts, src=head_mix_weights)
        return full_weights

    def forward(self, x_emb: torch.Tensor, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> Dict[str, torch.Tensor]:
        full_mix_weights = self._build_sparse_topk_weights(head_mix_weights, topk_experts)
        expert_outputs = torch.stack([expert(x_emb) for expert in self.experts], dim=1)
        mix = full_mix_weights.unsqueeze(-1).unsqueeze(-1)
        fused_seq = torch.sum(mix * expert_outputs, dim=1)
        fused_seq = self.fuse_norm(fused_seq)

        summary = torch.cat([fused_seq[:, -1, :], fused_seq.mean(dim=1)], dim=-1)
        point_pred = self.forecast_head(summary).view(x_emb.size(0), self.pred_len, self.out_dim)

        return {
            'mix_weights': full_mix_weights,
            'expert_sequences': expert_outputs,
            'fused_sequence': fused_seq,
            'point_pred': point_pred,
        }


class RetrievalBetaGate(nn.Module):
    def __init__(self, hidden_dim: int = 32, beta_min: float = 0.0, beta_max: float = 0.2):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _reduce_sims(sims: Optional[torch.Tensor], batch_size: int, channels: int, device: torch.device, dtype: torch.dtype):
        if sims is None:
            sim_mean = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            sim_max = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            sim_std = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            s = sims.unsqueeze(-1) if sims.dim() == 1 else sims.reshape(batch_size, -1)
            sim_mean = s.mean(dim=-1, keepdim=True)
            sim_max = s.max(dim=-1, keepdim=True).values
            sim_std = s.std(dim=-1, keepdim=True, unbiased=False)
        return sim_mean.expand(batch_size, channels), sim_max.expand(batch_size, channels), sim_std.expand(batch_size, channels)

    def forward(self, x_enc: torch.Tensor, base_pred: torch.Tensor, ret_pred: torch.Tensor, sims: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, _, c_x = x_enc.shape
        _, _, c_y = base_pred.shape

        x_mean = x_enc.mean(dim=1)
        x_std = x_enc.std(dim=1, unbiased=False)
        x_last = x_enc[:, -1, :]
        if c_x != c_y:
            x_mean = x_mean[:, :c_y]
            x_std = x_std[:, :c_y]
            x_last = x_last[:, :c_y]

        p_mean = base_pred.mean(dim=1)
        p_std = base_pred.std(dim=1, unbiased=False)
        r_mean = ret_pred.mean(dim=1)
        r_std = ret_pred.std(dim=1, unbiased=False)
        diff_mean = (base_pred - ret_pred).abs().mean(dim=1)
        sim_mean, sim_max, sim_std = self._reduce_sims(sims, batch_size, c_y, x_enc.device, x_enc.dtype)

        feat = torch.stack([
            x_mean, x_std, x_last,
            p_mean, p_std,
            r_mean, r_std,
            diff_mean,
            sim_mean, sim_max, sim_std,
        ], dim=-1)

        beta = torch.sigmoid(self.mlp(feat)).transpose(1, 2)
        return self.beta_min + (self.beta_max - self.beta_min) * beta


class ExtremeLSTMMemo(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        e_layers: int,
        d_layers: int,
        dec_in: int = 3,
        out_dim: int = 1,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dec_in = dec_in
        self.out_dim = out_dim
        self.dropout = self.config.dropout
        self.device = self.config.device

        self.num_experts = getattr(self.config, 'num_experts', 3)
        self.top_k_experts = min(getattr(self.config, 'top_k_experts', 2), self.num_experts)
        self.retrieval_num = getattr(self.config, 'retrieval_num', 2)
        self.retrieval_stride = 1

        self.retrieval_tau = getattr(self.config, 'retrieval_tau', 0.55)
        self.retrieval_alpha_max = getattr(self.config, 'retrieval_alpha_max', 0.02)
        self.retrieval_beta_hidden = getattr(self.config, 'retrieval_beta_hidden', 32)
        self.retrieval_beta_max = getattr(self.config, 'retrieval_beta_max', 0.20)
        self.retrieval_beta_reg = getattr(self.config, 'retrieval_beta_reg', 1e-4)

        include_last_value = bool(getattr(self.config, 'pretrain_include_last', True))
        state_dim = 4 if include_last_value else 3

        scales_cfg = getattr(self.config, 'state_prior_scales', (1, 4, 8, 16))
        if isinstance(scales_cfg, str):
            scales = tuple(int(s.strip()) for s in scales_cfg.split(',') if s.strip())
        else:
            scales = tuple(int(s) for s in scales_cfg)
        if len(scales) == 0:
            scales = (1,)

        self.state_prior = StudentTMixturePrior(
            num_components=self.num_experts,
            state_dim=state_dim,
            use_all_channels=bool(getattr(self.config, 'state_prior_use_all_channels', False)),
            include_last_value=include_last_value,
            scales=scales,
            include_seq_level=bool(getattr(self.config, 'state_prior_include_seq_level', True)),
            learnable_scale_weights=bool(getattr(self.config, 'state_prior_learnable_scale_weights', True)),
            min_scale=float(getattr(self.config, 'pretrain_min_scale', 1e-4)),
            min_df=float(getattr(self.config, 'pretrain_min_df', 2.1)),
            temperature=float(getattr(self.config, 'state_prior_temperature', 1.0)),
        )

        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.router = RouterFromEmbeddingPreTrain(
            num_experts=self.num_experts,
            hidden=getattr(self.config, 'router_hidden', 64),
            dropout=self.dropout,
        )
        self.backbone = BackboneMoE(
            d_model=d_model,
            pred_len=pred_len,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=min(self.dropout, 0.1),
            expert_layers=max(1, getattr(self.config, 'expert_layers', 1)),
        )
        self.beta_gate = RetrievalBetaGate(
            hidden_dim=self.retrieval_beta_hidden,
            beta_min=0.0,
            beta_max=self.retrieval_beta_max,
        )

        self.register_buffer('retrieval_gate_ready', torch.tensor(False, dtype=torch.bool), persistent=True)
        self.latest_aux_dict = {}

    def get_state_prior_parameters(self):
        return self.state_prior.parameters()

    def pretrain_state_prior_loss(self, x: torch.Tensor):
        prior_out = self.state_prior(x)
        loss = prior_out['pretrain_nll']
        aux = {
            'pretrain_nll': loss.detach(),
            'q_mean': prior_out['q'].detach().mean(dim=0),
            'mix_prob': prior_out['mix_prob'].detach(),
        }
        return loss, aux

    def freeze_state_prior(self):
        for p in self.state_prior.parameters():
            p.requires_grad = False

    def unfreeze_state_prior(self):
        for p in self.state_prior.parameters():
            p.requires_grad = True

    def freeze_backbone_for_gate(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
        for p in self.beta_gate.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def mark_gate_ready(self, ready: bool = True):
        self.retrieval_gate_ready.fill_(bool(ready))

    def compute_sample_level_balance_loss(self, router_logits: torch.Tensor):
        router_prob = torch.softmax(router_logits, dim=-1)
        load = router_prob.mean(dim=0)
        min_load = 0.2
        max_load = 0.5
        low_penalty = F.relu(min_load - load).pow(2)
        high_penalty = F.relu(load - max_load).pow(2)
        balance_loss = (low_penalty + high_penalty).sum()
        aux_dict = {'balance_loss': balance_loss.detach(), 'expert_load': load.detach()}
        return balance_loss, aux_dict

    def construct_index(self, num: int):
        self.keys = torch.zeros(num, self.seq_len, self.c_in, device=self.device)
        self.values = torch.zeros(num, self.pred_len, self.dec_in, device=self.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor):
        self.keys[index, :, :] = x_enc
        self.values[index, :, :] = y
        self.index += x_enc.size(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cosine_similarity(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if len(queries.shape) == 3:
            batch_size = queries.size(0)
            num_keys = keys.size(0)
            queries = queries.reshape(batch_size, -1)
            keys = keys.reshape(num_keys, -1)
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        if len(queries.shape) == 2:
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        raise ValueError(f'Unsupported query shape: {queries.shape}')

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        batch_size = x.shape[0]
        if self.index == 0:
            raise RuntimeError('Retrieval index has not been constructed yet.')
        k = min(self.retrieval_num, self.index)
        keys = self.keys[:self.index]
        values = self.values[:self.index]
        dis = self.cosine_similarity(x, keys)

        if self.training and index is not None:
            self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=x.device).unsqueeze(0)
            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[invalid_index < 0] = 0
            invalid_index[invalid_index >= self.index] = self.index - 1
            row_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
            dis[row_idx, invalid_index] = -100.0

        dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)
        sims = dis_topk
        probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)
        retrieved_values = values[indices_topk]
        output = torch.sum(probs_topk * retrieved_values, dim=1)
        return output, sims, 0

    def _forward_backbone(self, x: torch.Tensor):
        x_emb = self.enc_embedding(x)
        prior_out = self.state_prior(x)
        router_logits = self.router(prior_out['q'])
        router_prob = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
        head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        backbone_out = self.backbone(x_emb, head_mix_weights=head_mix_weights, topk_experts=topk_experts)
        backbone_out.update({
            'router_logits': router_logits,
            'router_prob': router_prob,
            'topk_experts': topk_experts,
            'topk_probs': head_mix_weights,
            'state_probs': prior_out['q'],
            'state_alpha': prior_out['mix_prob'],
            'state_pretrain_nll': prior_out['pretrain_nll'],
        })
        return backbone_out

    def _heuristic_fuse(self, point_pred: torch.Tensor, retrieval_pred: torch.Tensor, sims: torch.Tensor):
        sim_mean = sims.mean(dim=-1)
        dynamic_alpha = (sim_mean - self.retrieval_tau) / (1.0 - self.retrieval_tau + 1e-8)
        dynamic_alpha = dynamic_alpha.clamp(0.0, 1.0)
        dynamic_alpha = self.retrieval_alpha_max * dynamic_alpha
        dynamic_alpha = dynamic_alpha.view(-1, 1, 1)
        fused = (1 - dynamic_alpha) * point_pred + dynamic_alpha * retrieval_pred
        return fused, dynamic_alpha

    def _gate_fuse(self, x: torch.Tensor, point_pred: torch.Tensor, sample_ids: Optional[torch.Tensor]):
        retrieval_results, sims, _ = self.retrieval(x, sample_ids)
        retrieval_pred = retrieval_results[:, :, :self.out_dim]
        beta = self.beta_gate(x_enc=x, base_pred=point_pred.detach(), ret_pred=retrieval_pred.detach(), sims=sims)
        fused_point = (1.0 - beta) * point_pred + beta * retrieval_pred
        return fused_point, retrieval_pred, sims, beta

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        route_labels: Optional[torch.Tensor] = None,
        mode: str = 'train',
        return_aux: bool = False,
    ):
        if mode == 'gate_train':
            with torch.no_grad():
                out = self._forward_backbone(x=x)
        else:
            out = self._forward_backbone(x=x)

        point_pred = out['point_pred']
        total_aux_loss = point_pred.new_tensor(0.0)
        aux_dict = {}

        if mode in {'gate_train', 'gate_valid'}:
            fused_point, retrieval_pred, sims, beta = self._gate_fuse(x, point_pred, sample_ids)
            point_pred = fused_point
            total_aux_loss = total_aux_loss + self.retrieval_beta_reg * beta.mean()
            aux_dict['beta_mean'] = beta.mean().detach()
            aux_dict['beta_max'] = beta.max().detach()
            aux_dict['sim_mean'] = sims.mean().detach()
            out['retrieval_pred'] = retrieval_pred
            out['beta'] = beta
        elif mode == 'test' and hasattr(self, 'index') and self.index > 0:
            if bool(self.retrieval_gate_ready.item()):
                fused_point, retrieval_pred, sims, beta = self._gate_fuse(x, point_pred, sample_ids)
                point_pred = fused_point
                aux_dict['beta_mean'] = beta.mean().detach()
                aux_dict['beta_max'] = beta.max().detach()
                aux_dict['sim_mean'] = sims.mean().detach()
                out['retrieval_pred'] = retrieval_pred
                out['beta'] = beta
            else:
                retrieval_results, sims, _ = self.retrieval(x, sample_ids)
                retrieval_pred = retrieval_results[:, :, :self.out_dim]
                fused_point, dynamic_alpha = self._heuristic_fuse(point_pred, retrieval_pred, sims)
                point_pred = fused_point
                aux_dict['beta_mean'] = dynamic_alpha.mean().detach()
                aux_dict['beta_max'] = dynamic_alpha.max().detach()
                aux_dict['sim_mean'] = sims.mean().detach()
                out['retrieval_pred'] = retrieval_pred
                out['beta'] = dynamic_alpha

        if mode in {'train', 'valid'}:
            balance_loss, balance_aux_dict = self.compute_sample_level_balance_loss(out['router_logits'])
            total_aux_loss = total_aux_loss + 0.1 * balance_loss
            aux_dict.update(balance_aux_dict)

        out['point_pred'] = point_pred
        out['total_aux_loss'] = total_aux_loss
        aux_dict['total_aux_loss'] = total_aux_loss.detach()
        aux_dict['router_prob'] = out['router_prob'].detach()
        aux_dict['topk_experts'] = out['topk_experts'].detach()
        aux_dict['state_probs'] = out['state_probs'].detach()
        aux_dict['state_alpha'] = out['state_alpha'].detach()
        self.latest_aux_dict = aux_dict

        if return_aux:
            return out, total_aux_loss, aux_dict
        return out, total_aux_loss
