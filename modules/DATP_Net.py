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
        use_all_channels: bool = True,
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

        self.reset_parameters()

    def reset_parameters(self):
        # Break permutation symmetry at initialization to avoid uniform posterior collapse.
        # Also perturb scale/df slightly; if scale_raw and df_raw start exactly
        # identical, pairwise distance losses can have zero-distance gradients.
        with torch.no_grad():
            nn.init.normal_(self.mu, mean=0.0, std=0.05)
            nn.init.constant_(self.scale_raw, -1.0)
            nn.init.constant_(self.df_raw, 1.0)

            if self.num_components > 1:
                comp_offsets = torch.linspace(-0.5, 0.5, steps=self.num_components, device=self.mu.device)
                self.mu[:, 0] = self.mu[:, 0] + comp_offsets

                scale_offsets = torch.linspace(-0.12, 0.12, steps=self.num_components, device=self.scale_raw.device)
                df_offsets = torch.linspace(-0.08, 0.08, steps=self.num_components, device=self.df_raw.device)
                self.scale_raw.add_(scale_offsets.view(-1, 1))
                self.df_raw.add_(df_offsets.view(-1, 1))

            self.scale_raw.add_(0.01 * torch.randn_like(self.scale_raw))
            self.df_raw.add_(0.01 * torch.randn_like(self.df_raw))
            nn.init.normal_(self.mix_logits, mean=0.0, std=0.05)

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

    def get_component_params(self):
        """
        Return valid Student-T mixture parameters.

        mu:    [K, D]
        scale: [K, D], positive
        df:    [K, D], larger than min_df
        """
        scale = F.softplus(self.scale_raw.float()) + self.min_scale
        df = F.softplus(self.df_raw.float()) + self.min_df
        return self.mu.float(), scale, df

    def component_diversity_loss(
        self,
        mu_margin: float = 1.0,
        scale_margin: float = 0.3,
        df_margin: float = 0.2,
        scale_weight: float = 0.2,
        df_weight: float = 0.1,
        eps: float = 1e-6,
    ):
        """
        Softly separate different Student-T components.

        This loss only penalizes component pairs whose parameters are too close.
        The mixture likelihood is still responsible for keeping the summed
        distribution close to the training data.
        """
        if self.num_components <= 1:
            zero = self.mu.new_tensor(0.0)
            aux = {
                'mu_sep_loss': zero.detach(),
                'scale_sep_loss': zero.detach(),
                'df_sep_loss': zero.detach(),
                'mu_pair_dist_mean': zero.detach(),
                'scale_pair_dist_mean': zero.detach(),
                'df_pair_dist_mean': zero.detach(),
            }
            return zero, aux

        mu, scale, df = self.get_component_params()

        idx_i, idx_j = torch.triu_indices(
            self.num_components,
            self.num_components,
            offset=1,
            device=mu.device,
        )

        mu_i, mu_j = mu[idx_i], mu[idx_j]
        scale_i, scale_j = scale[idx_i], scale[idx_j]
        df_i, df_j = df[idx_i], df[idx_j]

        # Mean separation is normalized by the pooled component scale so that
        # dimensions with naturally larger scale do not dominate the distance.
        # The +eps is inside sqrt to avoid the undefined gradient of sqrt(0),
        # which is a common source of NaNs for pairwise distance losses.
        pooled_scale = torch.sqrt(0.5 * (scale_i.pow(2) + scale_j.pow(2)) + eps)
        mu_diff = (mu_i - mu_j) / pooled_scale
        mu_dist = torch.sqrt(mu_diff.pow(2).sum(dim=-1) + eps)

        # Scale and df are positive, so compare them in log-space.
        log_scale_diff = torch.log(scale_i + eps) - torch.log(scale_j + eps)
        log_scale_dist = torch.sqrt(log_scale_diff.pow(2).sum(dim=-1) + eps)

        log_df_diff = torch.log(df_i + eps) - torch.log(df_j + eps)
        log_df_dist = torch.sqrt(log_df_diff.pow(2).sum(dim=-1) + eps)

        mu_sep_loss = F.relu(mu_margin - mu_dist).pow(2).mean()
        scale_sep_loss = F.relu(scale_margin - log_scale_dist).pow(2).mean()
        df_sep_loss = F.relu(df_margin - log_df_dist).pow(2).mean()

        diversity_loss = (
            mu_sep_loss
            + scale_weight * scale_sep_loss
            + df_weight * df_sep_loss
        )

        aux = {
            'mu_sep_loss': mu_sep_loss.detach(),
            'scale_sep_loss': scale_sep_loss.detach(),
            'df_sep_loss': df_sep_loss.detach(),
            'mu_pair_dist_mean': mu_dist.mean().detach(),
            'scale_pair_dist_mean': log_scale_dist.mean().detach(),
            'df_pair_dist_mean': log_df_dist.mean().detach(),
        }

        return diversity_loss, aux

    def _component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D], output: [B, K]
        # Keep Student-T likelihood in fp32 even under AMP; lgamma/log/log1p
        # are numerically sensitive in fp16.
        z = z.float()
        mu = self.mu.float().unsqueeze(0)
        scale = F.softplus(self.scale_raw.float()).unsqueeze(0) + self.min_scale
        df = F.softplus(self.df_raw.float()).unsqueeze(0) + self.min_df
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
        # Force the probabilistic prior branch to fp32 for numerical stability.
        x_used = (x if self.use_all_channels else x[:, :, :1]).float()

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
        num_states: int,
        num_experts: int,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, q_prior: torch.Tensor) -> torch.Tensor:
        return self.net(q_prior)


class RouterFromEmbeddingFeatures(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        feat = torch.cat(
            [
                x_emb[:, -1, :],
                x_emb.mean(dim=1),
                x_emb.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.net(feat)


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

        self.num_experts = int(getattr(self.config, 'num_experts', 4))
        self.num_states = int(getattr(self.config, 'state_num', 0) or self.num_experts)
        if self.num_states < 1:
            raise ValueError(f'state_num must be >= 1, got {self.num_states}.')
        if self.num_experts < 1:
            raise ValueError(f'num_experts must be >= 1, got {self.num_experts}.')
        self.top_k_experts = min(int(getattr(self.config, 'top_k_experts', 2)), self.num_experts)
        self.use_retrieval = bool(getattr(self.config, 'use_retrieval', True))
        self.use_state_prior = bool(getattr(self.config, 'use_state_prior', True))
        self.retrieval_num = int(getattr(self.config, 'retrieval_num', 2))
        self.retrieval_stride = 1

        self.retrieval_tau = getattr(self.config, 'retrieval_tau', 0.55)
        self.retrieval_alpha_max = getattr(self.config, 'retrieval_alpha_max', 0.02)
        self.retrieval_beta_hidden = getattr(self.config, 'retrieval_beta_hidden', 32)
        self.retrieval_beta_max = getattr(self.config, 'retrieval_beta_max', 0.1)
        self.retrieval_beta_reg = getattr(self.config, 'retrieval_beta_reg', 1e-4)
        self.state_balance_weight = float(getattr(self.config, 'state_balance_weight', 0.02))
        self.state_dom_cap = float(getattr(self.config, 'state_dom_cap', 0.8))

        # Component-level Student-T diversity constraints.
        # These are soft constraints: NLL still keeps the mixture distribution
        # covering the training samples, while these terms prevent different
        # components from becoming nearly identical.
        # self.state_diversity_weight = float(getattr(self.config, 'state_diversity_weight', 0.0))
        # self.state_assignment_entropy_weight = float(getattr(self.config, 'state_assignment_entropy_weight', 0.0))
        self.state_diversity_weight = float(getattr(self.config, 'state_diversity_weight', 0.001))
        self.state_assignment_entropy_weight = float(getattr(self.config, 'state_assignment_entropy_weight', 0.0005))
        self.state_mu_margin = float(getattr(self.config, 'state_mu_margin', 1.0))
        self.state_scale_margin = float(getattr(self.config, 'state_scale_margin', 0.3))
        self.state_df_margin = float(getattr(self.config, 'state_df_margin', 0.2))
        self.state_scale_diversity_weight = float(getattr(self.config, 'state_scale_diversity_weight', 0.2))
        self.state_df_diversity_weight = float(getattr(self.config, 'state_df_diversity_weight', 0.1))
        self.router_balance_weight = float(getattr(self.config, 'router_balance_weight', 0.1))
        self.topk_coverage_weight = float(getattr(self.config, 'topk_coverage_weight', 0.25))
        self.topk_min_usage = float(getattr(self.config, 'topk_min_usage', 0.12))
        self.router_temperature = float(getattr(self.config, 'router_temperature', 1.0))
        self.router_train_noise_std = float(getattr(self.config, 'router_train_noise_std', 0.0))
        self.ensure_all_experts_in_topk = bool(getattr(self.config, 'ensure_all_experts_in_topk', True))

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
            num_components=self.num_states,
            state_dim=state_dim,
            use_all_channels=bool(getattr(self.config, 'state_prior_use_all_channels', True)),
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
            num_states=self.num_states,
            num_experts=self.num_experts,
            hidden=getattr(self.config, 'router_hidden', 64),
            dropout=self.dropout,
        )
        self.learned_router = RouterFromEmbeddingFeatures(
            d_model=d_model,
            num_experts=self.num_experts,
            hidden=getattr(self.config, 'learned_router_hidden', getattr(self.config, 'router_hidden', 64)),
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
        if not self.use_state_prior:
            zero = x.new_tensor(0.0)
            uniform = torch.full(
                (self.num_states,),
                1.0 / self.num_states,
                device=x.device,
                dtype=x.dtype,
            )
            aux = {
                'pretrain_nll': zero.detach(),
                'pretrain_total_loss': zero.detach(),
                'q_mean': uniform.detach(),
                'mix_prob': uniform.detach(),
                'balance_kl': zero.detach(),
                'dominant_penalty': zero.detach(),
                'diversity_loss': zero.detach(),
                'assignment_entropy': zero.detach(),
            }
            return zero, aux

        # Student-t prior pretraining uses only valid value channels by default.
        # This prevents the prior from learning missingness patterns instead of
        # traffic-state patterns.
        prior_x = self._extract_state_prior_input(x)
        prior_out = self.state_prior(prior_x)
        eps = 1e-8
        q = prior_out['q']
        q_mean = q.mean(dim=0)

        # 1) Mixture NLL: keep the summed Student-T mixture distribution
        # covering the training state vectors.
        nll_loss = prior_out['pretrain_nll']

        # 2) Batch-level state usage balance: prevent all samples from collapsing
        # into one component.
        uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())
        balance_kl = torch.sum(q_mean * (torch.log(q_mean + eps) - torch.log(uniform + eps)))
        dominant_penalty = F.relu(q_mean.max() - self.state_dom_cap).pow(2)

        # 3) Component-level diversity: separate mu / scale / df among different
        # Student-T components.
        diversity_loss, diversity_aux = self.state_prior.component_diversity_loss(
            mu_margin=self.state_mu_margin,
            scale_margin=self.state_scale_margin,
            df_margin=self.state_df_margin,
            scale_weight=self.state_scale_diversity_weight,
            df_weight=self.state_df_diversity_weight,
        )

        # 4) Sample-level confidence: mildly reduce assignment entropy so each
        # sample has a clearer state responsibility. Keep this weight small.
        assignment_entropy = -(q * torch.log(q + eps)).sum(dim=-1).mean()

        loss = (
            nll_loss
            + self.state_balance_weight * (balance_kl + dominant_penalty)
            + self.state_diversity_weight * diversity_loss
            + self.state_assignment_entropy_weight * assignment_entropy
        )

        aux = {
            'pretrain_nll': nll_loss.detach(),
            'pretrain_total_loss': loss.detach(),
            'q_mean': q_mean.detach(),
            'mix_prob': prior_out['mix_prob'].detach(),
            'balance_kl': balance_kl.detach(),
            'dominant_penalty': dominant_penalty.detach(),
            'diversity_loss': diversity_loss.detach(),
            'assignment_entropy': assignment_entropy.detach(),
        }
        aux.update(diversity_aux)

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
        self.retrieval_gate_ready.fill_(bool(ready) and self.use_retrieval)

    def compute_sample_level_balance_loss(self, router_logits: torch.Tensor):
        router_prob = torch.softmax(router_logits, dim=-1)
        load = router_prob.mean(dim=0)
        if load.numel() <= 1:
            balance_loss = router_logits.sum() * 0.0
            aux_dict = {'balance_loss': balance_loss.detach(), 'expert_load': load.detach()}
            return balance_loss, aux_dict

        min_load = 0.2
        max_load = 0.5
        low_penalty = F.relu(min_load - load).pow(2)
        high_penalty = F.relu(load - max_load).pow(2)
        balance_loss = (low_penalty + high_penalty).sum()
        aux_dict = {'balance_loss': balance_loss.detach(), 'expert_load': load.detach()}
        return balance_loss, aux_dict

    def compute_topk_coverage_loss(self, router_prob: torch.Tensor, topk_experts: torch.Tensor):
        if router_prob.size(-1) <= 1:
            zero = router_prob.sum() * 0.0
            return zero, {'topk_coverage_loss': zero.detach(), 'topk_usage': router_prob.mean(dim=0).detach()}

        selected = F.one_hot(topk_experts, num_classes=self.num_experts).sum(dim=1).clamp(max=1).float()
        topk_usage = selected.mean(dim=0)
        soft_load = router_prob.mean(dim=0)
        uniform = torch.full_like(soft_load, 1.0 / soft_load.numel())

        eps = 1e-8
        soft_balance = torch.sum(soft_load * (torch.log(soft_load + eps) - torch.log(uniform + eps)))
        floor_penalty = F.relu(self.topk_min_usage - soft_load).pow(2).sum()

        loss = soft_balance + floor_penalty
        aux_dict = {
            'topk_coverage_loss': loss.detach(),
            'topk_usage': topk_usage.detach(),
            'soft_expert_load': soft_load.detach(),
        }
        return loss, aux_dict

    def _select_topk_experts(self, router_logits: torch.Tensor):
        if self.router_temperature > 0:
            selection_logits = router_logits / self.router_temperature
        else:
            selection_logits = router_logits

        if self.training and self.router_train_noise_std > 0:
            selection_logits = selection_logits + torch.randn_like(selection_logits) * self.router_train_noise_std

        router_prob = torch.softmax(selection_logits, dim=-1)
        topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)

        if not self.ensure_all_experts_in_topk:
            head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            return router_prob, topk_experts, head_mix_weights

        batch_size = router_prob.size(0)
        if batch_size <= 0 or self.top_k_experts >= self.num_experts:
            head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            return router_prob, topk_experts, head_mix_weights

        # Quota repair: keep normal router probabilities, but ensure every expert
        # appears in the batch-level Top-K set when the batch is large enough.
        repaired = topk_experts.clone()
        repaired_probs = topk_probs.clone()
        for expert_id in range(self.num_experts):
            contains = (repaired == expert_id).any(dim=1)
            if contains.any():
                continue

            eligible = ~contains
            if not eligible.any():
                continue

            replace_slot = repaired_probs.argmin(dim=1)
            row_ids = torch.arange(batch_size, device=router_prob.device)
            current_min_prob = repaired_probs[row_ids, replace_slot]
            candidate_prob = router_prob[:, expert_id]
            cost = current_min_prob - candidate_prob
            cost = torch.where(eligible, cost, cost.new_full(cost.shape, float('inf')))
            chosen = torch.argmin(cost)
            slot = replace_slot[chosen]
            repaired[chosen, slot] = expert_id
            repaired_probs[chosen, slot] = candidate_prob[chosen]

        order = torch.argsort(repaired_probs, dim=1, descending=True)
        topk_experts = torch.gather(repaired, dim=1, index=order)
        topk_probs = torch.gather(repaired_probs, dim=1, index=order)
        head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return router_prob, topk_experts, head_mix_weights

    def construct_index(self, num: int):
        if not self.use_retrieval:
            self.index = 0
            return
        self.keys = torch.zeros(num, self.seq_len, self.c_in, device=self.device)
        self.values = torch.zeros(num, self.pred_len, self.dec_in, device=self.device)
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor):
        if not self.use_retrieval:
            return
        self.keys[index, :, :] = x_enc
        self.values[index, :, :] = y
        self.index += x_enc.size(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _split_missing_aware_input(self, x: torch.Tensor):
        """
        Split missing-aware input channels.

        Expected feature order:
            [diff_norm, diff_mask, second_diff_raw, second_diff_mask, raw, raw_mask]

        For retrieval similarity, only value channels are compared:
            [diff_norm, second_diff_raw, raw]

        The corresponding validity masks are:
            [diff_mask, second_diff_mask, raw_mask]

        If the input is not in 6-group missing-aware format, fall back to the
        original behavior with an all-one mask.
        """
        if x.dim() != 3:
            return x, torch.ones_like(x)

        groups = int(getattr(self.config, 'missing_aware_groups', 6))
        if groups != 6 or x.size(-1) % 6 != 0:
            return x, torch.ones_like(x)

        c = x.size(-1) // 6

        diff = x[..., 0:c]
        diff_mask = x[..., c:2 * c].clamp(0.0, 1.0)

        second_diff = x[..., 2 * c:3 * c]
        second_mask = x[..., 3 * c:4 * c].clamp(0.0, 1.0)

        raw = x[..., 4 * c:5 * c]
        raw_mask = x[..., 5 * c:6 * c].clamp(0.0, 1.0)

        values = torch.cat([diff, second_diff, raw], dim=-1)
        masks = torch.cat([diff_mask, second_mask, raw_mask], dim=-1)

        return values, masks

    def _extract_state_prior_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build the input used by the Student-t state prior.

        The full forecasting backbone still receives all missing-aware channels:
            [diff_norm, diff_mask, second_diff_raw, second_diff_mask, raw, raw_mask]

        However, the Student-t prior should model traffic-state statistics rather
        than missingness patterns. Therefore, by default it only receives valid
        value channels:
            [diff_norm * diff_mask,
             second_diff_raw * second_diff_mask,
             raw * raw_mask]

        If config.state_prior_use_value_only=False, this function falls back to
        the old behavior and returns the full input x.
        """
        use_value_only = bool(getattr(self.config, 'state_prior_use_value_only', True))
        if not use_value_only:
            return x

        values, masks = self._split_missing_aware_input(x)
        return values * masks

    def cosine_similarity(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Missing-aware cosine similarity for retrieval.

        Similarity is computed on value channels only and only on dimensions
        that are valid in both the query and the key. For flattened value
        vectors q and k with masks m^q and m^k, the pairwise similarity is:

            sum_i q_i k_i m_i^q m_i^k
            -----------------------------------------------
            sqrt(sum_i q_i^2 m_i^q m_i^k)
            sqrt(sum_i k_i^2 m_i^q m_i^k) + eps

        This avoids two windows being considered similar just because they share
        missing placeholders or mask patterns.
        """
        eps = 1e-8

        if len(queries.shape) == 3:
            batch_size = queries.size(0)
            num_keys = keys.size(0)

            q_values, q_masks = self._split_missing_aware_input(queries)
            k_values, k_masks = self._split_missing_aware_input(keys)

            q_values = q_values.reshape(batch_size, -1)
            q_masks = q_masks.reshape(batch_size, -1)
            k_values = k_values.reshape(num_keys, -1)
            k_masks = k_masks.reshape(num_keys, -1)

            numerator = torch.matmul(q_values * q_masks, (k_values * k_masks).t())
            q_sq = torch.matmul(q_values.pow(2) * q_masks, k_masks.t())
            k_sq = torch.matmul(q_masks, (k_values.pow(2) * k_masks).t())
            denom = torch.sqrt(q_sq.clamp_min(0.0)) * torch.sqrt(k_sq.clamp_min(0.0)) + eps

            sim = numerator / denom

            common_valid = torch.matmul(q_masks, k_masks.t())
            sim = torch.where(common_valid > 0, sim, sim.new_full(sim.shape, -1.0))

            return sim

        if len(queries.shape) == 2:
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())

        raise ValueError(f'Unsupported query shape: {queries.shape}')

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        if not self.use_retrieval:
            raise RuntimeError('Retrieval is disabled by config.use_retrieval=False.')
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

        if self.use_state_prior:
            # The backbone uses the full missing-aware input, while the Student-t
            # prior only uses valid value channels by default.
            prior_x = self._extract_state_prior_input(x)
            prior_out = self.state_prior(prior_x)
            router_logits = self.router(prior_out['q'])
            state_probs = prior_out['q']
            state_z = prior_out['z']
            state_alpha = prior_out['mix_prob']
            state_pretrain_nll = prior_out['pretrain_nll']
            router_source = 'student_t_prior'
        else:
            router_logits = self.learned_router(x_emb)
            router_prob_for_aux = torch.softmax(router_logits, dim=-1)
            state_probs = router_prob_for_aux
            state_z = torch.cat(
                [
                    x_emb[:, -1, :],
                    x_emb.mean(dim=1),
                    x_emb.std(dim=1, unbiased=False),
                ],
                dim=-1,
            )
            state_alpha = router_prob_for_aux.mean(dim=0)
            state_pretrain_nll = x_emb.new_tensor(0.0)
            router_source = 'embedding_router'

        router_prob, topk_experts, head_mix_weights = self._select_topk_experts(router_logits)

        backbone_out = self.backbone(x_emb, head_mix_weights=head_mix_weights, topk_experts=topk_experts)
        backbone_out.update({
            'router_logits': router_logits,
            'router_prob': router_prob,
            'topk_experts': topk_experts,
            'topk_probs': head_mix_weights,
            'state_probs': state_probs,
            'state_z': state_z,
            'state_alpha': state_alpha,
            'state_pretrain_nll': state_pretrain_nll,
            'router_source': router_source,
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
        dec_mark: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
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

        if not self.use_retrieval:
            aux_dict['retrieval_disabled'] = point_pred.new_tensor(1.0).detach()
        elif mode in {'gate_train', 'gate_valid'}:
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
                print("进入检索状态。。。。。。。。。。。。。。。。。。。。。。。。。。。。。")
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
            coverage_loss, coverage_aux_dict = self.compute_topk_coverage_loss(out['router_prob'], out['topk_experts'])
            total_aux_loss = total_aux_loss + self.router_balance_weight * balance_loss + self.topk_coverage_weight * coverage_loss
            aux_dict.update(balance_aux_dict)
            aux_dict.update(coverage_aux_dict)

            if self.use_state_prior:
                # Keep prior states from collapsing during backbone training.
                eps = 1e-8
                q = out['state_probs']
                q_mean = q.mean(dim=0)
                q_uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())
                state_balance_loss = torch.sum(q_mean * (torch.log(q_mean + eps) - torch.log(q_uniform + eps)))
                state_dom_loss = F.relu(q_mean.max() - self.state_dom_cap).pow(2)

                # Keep Student-T components distinguishable during backbone training
                # as well; otherwise fine-tuning may make the pre-trained prior drift
                # back toward overlapping components.
                state_diversity_loss, state_diversity_aux = self.state_prior.component_diversity_loss(
                    mu_margin=self.state_mu_margin,
                    scale_margin=self.state_scale_margin,
                    df_margin=self.state_df_margin,
                    scale_weight=self.state_scale_diversity_weight,
                    df_weight=self.state_df_diversity_weight,
                )
                state_assignment_entropy = -(q * torch.log(q + eps)).sum(dim=-1).mean()

                total_aux_loss = (
                    total_aux_loss
                    + self.state_balance_weight * (state_balance_loss + state_dom_loss)
                    + self.state_diversity_weight * state_diversity_loss
                    + self.state_assignment_entropy_weight * state_assignment_entropy
                )

                aux_dict['state_balance_loss'] = state_balance_loss.detach()
                aux_dict['state_dom_loss'] = state_dom_loss.detach()
                aux_dict['state_qmax'] = q_mean.max().detach()
                aux_dict['state_diversity_loss'] = state_diversity_loss.detach()
                aux_dict['state_assignment_entropy'] = state_assignment_entropy.detach()
                aux_dict.update(state_diversity_aux)
            else:
                aux_dict['state_prior_disabled'] = point_pred.new_tensor(1.0).detach()

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
