from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding
from layers.att.cross_attention import CrossAttention


# =========================================================
# 1. 同构专家：所有 expert 结构相同，差异只来自参数与路由分配
# =========================================================
class HomogeneousPointExpert(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        x = self.norm(residual + x)
        return x


# =========================================================
# 2. Student-T 原型状态打分器（方案 A）
#    输出 q_normal, q_mid, q_extreme，作为 router 辅助特征
# =========================================================
class StudentTStatePrior(nn.Module):
    def __init__(
        self,
        use_all_channels: bool = False,
        temperature: float = 1.0,
        learnable_prototypes: bool = False,
        normal_scale: float = 0.5,
        mid_scale: float = 1.0,
        extreme_scale: float = 2.0,
        normal_df: float = 20.0,
        mid_df: float = 6.0,
        extreme_df: float = 2.5,
        min_scale: float = 1e-4,
        min_df: float = 2.1,
    ):
        super().__init__()
        self.use_all_channels = use_all_channels
        self.temperature = temperature
        self.min_scale = min_scale
        self.min_df = min_df

        init_mu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        init_scale = torch.tensor([normal_scale, mid_scale, extreme_scale], dtype=torch.float32)
        init_df = torch.tensor([normal_df, mid_df, extreme_df], dtype=torch.float32)

        if learnable_prototypes:
            self.proto_mu = nn.Parameter(init_mu)
            self.proto_scale_raw = nn.Parameter(torch.log(torch.exp(init_scale - min_scale) - 1.0))
            self.proto_df_raw = nn.Parameter(torch.log(torch.exp(init_df - min_df) - 1.0))
        else:
            self.register_buffer('proto_mu', init_mu, persistent=True)
            self.register_buffer('proto_scale', init_scale, persistent=True)
            self.register_buffer('proto_df', init_df, persistent=True)
            self.proto_scale_raw = None
            self.proto_df_raw = None

    def _get_proto_params(self):
        if self.proto_scale_raw is None:
            mu = self.proto_mu
            scale = self.proto_scale
            df = self.proto_df
        else:
            mu = self.proto_mu
            scale = F.softplus(self.proto_scale_raw) + self.min_scale
            df = F.softplus(self.proto_df_raw) + self.min_df
        return mu, scale, df

    def _student_t_log_prob(self, x: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, df: torch.Tensor):
        # x: [B, N], mu/scale/df: [3]
        x = x.unsqueeze(1)          # [B, 1, N]
        mu = mu.view(1, -1, 1)      # [1, 3, 1]
        scale = scale.view(1, -1, 1)
        df = df.view(1, -1, 1)

        z = (x - mu) / scale
        log_norm = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * (torch.log(df) + torch.log(torch.as_tensor(torch.pi, device=x.device, dtype=x.dtype)))
            - torch.log(scale)
        )
        log_kernel = -((df + 1.0) / 2.0) * torch.log1p((z ** 2) / df)
        return log_norm + log_kernel  # [B, 3, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: q in [B, 3], 分别对应 normal / mid / extreme
        """
        if self.use_all_channels:
            x_used = x
        else:
            x_used = x[:, :, :1]  # 默认仅看第 1 个通道（通常就是标准化后的差分主目标）

        B = x_used.size(0)
        x_flat = x_used.reshape(B, -1)
        mu, scale, df = self._get_proto_params()
        log_prob = self._student_t_log_prob(x_flat, mu, scale, df)  # [B, 3, N]
        score = log_prob.sum(dim=-1)                                # [B, 3]
        q = torch.softmax(score / self.temperature, dim=-1)
        return q


# =========================================================
# 3. Router：方案 A，把 [std, maxabs, last, q_normal, q_mid, q_extreme] 拼起来
# =========================================================
class SampleRouterFromX(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_experts: int,
        hidden: int = 32,
        dropout: float = 0.0,
        state_prior_use_all_channels: bool = False,
        state_prior_temperature: float = 1.0,
        state_prior_learnable: bool = False,
    ):
        super().__init__()
        self.state_prior = StudentTStatePrior(
            use_all_channels=state_prior_use_all_channels,
            temperature=state_prior_temperature,
            learnable_prototypes=state_prior_learnable,
        )

        in_dim = 3 * c_in + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor):
        std = x.std(dim=1, unbiased=False)   # [B, C]
        max_abs = x.abs().amax(dim=1)        # [B, C]
        last = x[:, -1, :]                   # [B, C]
        q = self.state_prior(x)              # [B, 3]

        feat = torch.cat([std, max_abs, last, q], dim=-1)
        logits = self.net(feat)
        return logits, q


# =========================================================
# 4. 点预测 MoE Head（同构专家版）
# =========================================================
class PointMoEHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_experts: int = 3,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.experts = nn.ModuleList([
            HomogeneousPointExpert(d_model=d_model, hidden=expert_hidden or d_model, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.point_heads = nn.ModuleList([nn.Linear(d_model, out_dim) for _ in range(num_experts)])

    def _build_sparse_topk_weights(self, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> torch.Tensor:
        B, _ = head_mix_weights.shape
        full_weights = torch.zeros(B, self.num_experts, device=head_mix_weights.device, dtype=head_mix_weights.dtype)
        full_weights.scatter_(dim=1, index=topk_experts, src=head_mix_weights)
        return full_weights

    def forward(self, x: torch.Tensor, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> Dict[str, torch.Tensor]:
        full_mix_weights = self._build_sparse_topk_weights(head_mix_weights, topk_experts)  # [B, E]

        points = []
        for e in range(self.num_experts):
            feat_e = self.experts[e](x)            # [B, H, D]
            point_e = self.point_heads[e](feat_e)  # [B, H, O]
            points.append(point_e)

        point_all = torch.stack(points, dim=1)  # [B, E, H, O]
        point_pred = torch.sum(full_mix_weights[:, :, None, None] * point_all, dim=1)  # [B, H, O]

        return {
            'mix_weights': full_mix_weights,
            'expert_points': point_all,
            'point_pred': point_pred,
        }


# =========================================================
# 5. 检索融合门控：保留原思想
# =========================================================
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

    def _reduce_sims(self, sims: Optional[torch.Tensor], B: int, C: int, device: torch.device, dtype: torch.dtype):
        if sims is None:
            sim_mean = torch.zeros(B, 1, device=device, dtype=dtype)
            sim_max = torch.zeros(B, 1, device=device, dtype=dtype)
            sim_std = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            if sims.dim() == 1:
                s = sims.unsqueeze(-1)
            else:
                s = sims.reshape(B, -1)
            sim_mean = s.mean(dim=-1, keepdim=True)
            sim_max = s.max(dim=-1, keepdim=True).values
            sim_std = s.std(dim=-1, keepdim=True, unbiased=False)
        return sim_mean.expand(B, C), sim_max.expand(B, C), sim_std.expand(B, C)

    def forward(self, x_enc: torch.Tensor, base_pred: torch.Tensor, ret_pred: torch.Tensor, sims: Optional[torch.Tensor]) -> torch.Tensor:
        B, _, C_x = x_enc.shape
        _, _, C_y = base_pred.shape

        x_mean = x_enc.mean(dim=1)
        x_std = x_enc.std(dim=1, unbiased=False)
        x_last = x_enc[:, -1, :]
        if C_x != C_y:
            x_mean = x_mean[:, :C_y]
            x_std = x_std[:, :C_y]
            x_last = x_last[:, :C_y]

        p_mean = base_pred.mean(dim=1)
        p_std = base_pred.std(dim=1, unbiased=False)
        r_mean = ret_pred.mean(dim=1)
        r_std = ret_pred.std(dim=1, unbiased=False)
        diff_mean = (base_pred - ret_pred).abs().mean(dim=1)
        sim_mean, sim_max, sim_std = self._reduce_sims(sims, B, C_y, x_enc.device, x_enc.dtype)

        feat = torch.stack([
            x_mean, x_std, x_last,
            p_mean, p_std,
            r_mean, r_std,
            diff_mean,
            sim_mean, sim_max, sim_std,
        ], dim=-1)

        beta = torch.sigmoid(self.mlp(feat))
        beta = beta.transpose(1, 2)
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta
        return beta


# =========================================================
# 6. 主模型：点预测版 + 分布状态辅助路由 + 保留检索库
# =========================================================
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
        self.retrieval_num = getattr(self.config, 'retrieval_num', 2)
        self.top_k_experts = min(getattr(self.config, 'top_k_experts', 2), self.num_experts)
        self.retrieval_stride = 1

        self.retrieval_tau = getattr(self.config, 'retrieval_tau', 0.55)
        self.retrieval_alpha_max = getattr(self.config, 'retrieval_alpha_max', 0.02)
        self.retrieval_beta_hidden = getattr(self.config, 'retrieval_beta_hidden', 32)
        self.retrieval_beta_max = getattr(self.config, 'retrieval_beta_max', 0.20)
        self.retrieval_beta_reg = getattr(self.config, 'retrieval_beta_reg', 1e-4)

        self.register_buffer('retrieval_gate_ready', torch.tensor(False, dtype=torch.bool), persistent=True)

        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.dec_embedding = DataEmbedding(c_in=dec_in, d_model=d_model, dropout=self.dropout)

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=e_layers,
            batch_first=True,
            dropout=self.dropout if e_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=d_layers,
            batch_first=True,
            dropout=self.dropout if d_layers > 1 else 0.0,
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        self.router = SampleRouterFromX(
            c_in=c_in,
            num_experts=self.num_experts,
            hidden=getattr(self.config, 'router_hidden', 32),
            dropout=self.dropout,
            state_prior_use_all_channels=getattr(self.config, 'state_prior_use_all_channels', False),
            state_prior_temperature=getattr(self.config, 'state_prior_temperature', 1.0),
            state_prior_learnable=getattr(self.config, 'state_prior_learnable', False),
        )

        self.moe_head = PointMoEHead(
            d_model=d_model,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=min(self.dropout, 0.1),
            expert_hidden=getattr(self.config, 'expert_hidden', d_model),
        )

        self.beta_gate = RetrievalBetaGate(
            hidden_dim=self.retrieval_beta_hidden,
            beta_min=0.0,
            beta_max=self.retrieval_beta_max,
        )

        self.latest_aux_dict = {}

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
            B = queries.size(0)
            N = keys.size(0)
            queries = queries.reshape(B, -1)
            keys = keys.reshape(N, -1)
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        if len(queries.shape) == 2:
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        raise ValueError(f'Unsupported query shape: {queries.shape}')

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        bs = x.shape[0]
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
            row_idx = torch.arange(bs, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
            dis[row_idx, invalid_index] = -100.0

        dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)
        sims = dis_topk
        probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)
        retrieved_values = values[indices_topk]
        output = torch.sum(probs_topk * retrieved_values, dim=1)
        return output, sims, 0

    def _forward_backbone(self, x: torch.Tensor, dec_input: Optional[torch.Tensor] = None):
        router_logits, state_probs = self.router(x)
        router_prob = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
        head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        x_emb_hist = self.enc_embedding(x)
        dec_emb = self.dec_embedding(dec_input)

        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)
        dec_out, _ = self.decoder(dec_emb, (h_n, c_n))
        dec_out = dec_out[:, -self.pred_len:, :]

        ctx, _ = self.xattn(dec_out, enc_out, enc_out)
        fused = torch.cat([dec_out, ctx], dim=-1)
        fused = self.fuse_proj(fused)
        final_shared = self.post_norm(fused)

        moe_out = self.moe_head(final_shared, head_mix_weights=head_mix_weights, topk_experts=topk_experts)
        moe_out.update({
            'router_logits': router_logits,
            'router_prob': router_prob,
            'topk_experts': topk_experts,
            'topk_probs': head_mix_weights,
            'state_probs': state_probs,
        })
        return moe_out

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
                out = self._forward_backbone(x=x, dec_input=dec_input)
        else:
            out = self._forward_backbone(x=x, dec_input=dec_input)

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
        self.latest_aux_dict = aux_dict

        if return_aux:
            return out, total_aux_loss, aux_dict
        return out, total_aux_loss
