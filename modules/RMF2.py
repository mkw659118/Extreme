from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding
from layers.att.cross_attention import CrossAttention


# =========================================================
# 1. 异质专家：更适合 sample-level routing + top2
#    expert0: 轻量线性残差
#    expert1: bottleneck MLP 残差
#    expert2: temporal conv 残差
# =========================================================
class LinearResidualExpert(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.drop(self.act(self.fc(x)))
        x = self.norm(residual + x)
        return x


class BottleneckMLPExpert(nn.Module):
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


class TemporalConvExpert(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model
        )
        self.pw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, pred_len, d_model]
        residual = x
        x = x.transpose(1, 2)          # [B, d_model, pred_len]
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.drop(x)
        x = x.transpose(1, 2)          # [B, pred_len, d_model]
        x = self.norm(residual + x)
        return x


# =========================================================
# 2. Router：只基于输入样本统计量进行路由
# =========================================================
class SampleRouterFromX(nn.Module):
    """
    输入:
        x: [B, L, C]
    输出:
        logits: [B, num_experts]
    """
    def __init__(self, c_in: int, num_experts: int, hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        in_dim = 3 * c_in

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=1, unbiased=False)   # [B, C]
        max_abs = x.abs().amax(dim=1)        # [B, C]
        last = x[:, -1, :]                   # [B, C]

        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        logits = self.net(feat)                         # [B, E]
        return logits


# =========================================================
# 3. 标准 MoE Head（异质专家 + top2 + head residual）
# =========================================================
class StandardMoEHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_experts: int = 3,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k

        if num_experts != 3:
            raise ValueError("This hetero expert version currently assumes num_experts=3.")

        self.experts = nn.ModuleList([
            LinearResidualExpert(d_model=d_model, dropout=dropout),                      # expert 0
            BottleneckMLPExpert(d_model=d_model, hidden=d_model, dropout=dropout),       # expert 1
            TemporalConvExpert(d_model=d_model, kernel_size=3, dropout=dropout),         # expert 2
        ])

        self.head_norm = nn.RMSNorm(d_model)
        self.final_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x: torch.Tensor,             # [B, pred_len, d_model]
        router_probs: torch.Tensor,  # [B, K]
        topk_experts: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        B, _, _ = x.shape
        moe_delta = torch.zeros_like(x)  # 聚合 top-k 专家的修正量

        for k in range(self.top_k):
            expert_idx = topk_experts[:, k]             # [B]
            weight = router_probs[:, k].view(B, 1, 1)  # [B, 1, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue

                expert_feat = self.experts[e](x[mask])      # [mask_B, pred_len, d_model]
                moe_delta[mask] += expert_feat * weight[mask]

        moe_out = self.head_norm(x + moe_delta)
        final_out = self.final_proj(moe_out)
        return final_out


# =========================================================
# 4. 检索融合门控
# =========================================================
class RetrievalBetaGate(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        beta_min: float = 0.0,
        beta_max: float = 0.2,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

        # 每个通道 11 维特征 -> 1 个 beta
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _reduce_sims(
        self,
        sims: Optional[torch.Tensor],
        B: int,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if sims is None:
            sim_mean = torch.zeros(B, 1, device=device, dtype=dtype)
            sim_max = torch.zeros(B, 1, device=device, dtype=dtype)
            sim_std = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            if sims.dim() == 1:
                s = sims.unsqueeze(-1)  # [B, 1]
            else:
                s = sims.reshape(B, -1)

            sim_mean = s.mean(dim=-1, keepdim=True)
            sim_max = s.max(dim=-1, keepdim=True).values
            sim_std = s.std(dim=-1, keepdim=True, unbiased=False)

        return (
            sim_mean.expand(B, C),
            sim_max.expand(B, C),
            sim_std.expand(B, C),
        )

    def forward(
        self,
        x_enc: torch.Tensor,        # [B, L, C]
        base_pred: torch.Tensor,    # [B, pred_len, out_dim]
        ret_pred: torch.Tensor,     # [B, pred_len, out_dim]
        sims: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, C_x = x_enc.shape
        _, _, C_y = base_pred.shape

        device = x_enc.device
        dtype = x_enc.dtype

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

        sim_mean, sim_max, sim_std = self._reduce_sims(
            sims=sims,
            B=B,
            C=C_y,
            device=device,
            dtype=dtype,
        )

        feat = torch.stack([
            x_mean, x_std, x_last,
            p_mean, p_std,
            r_mean, r_std,
            diff_mean,
            sim_mean, sim_max, sim_std,
        ], dim=-1)  # [B, C_y, 11]

        beta = torch.sigmoid(self.mlp(feat))  # [B, C_y, 1]
        beta = beta.transpose(1, 2)           # [B, 1, C_y]
        beta = self.beta_min + (self.beta_max - self.beta_min) * beta
        return beta


# =========================================================
# 5. 主模型：ExtremeLSTMMemo
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

        # ---------------- experts / retrieval ----------------
        self.num_experts = 3
        self.retrieval_num = getattr(self.config, "retrieval_num", 2)
        self.top_k_experts = 2
        self.retrieval_stride = 1

        self.retrieval_tau = getattr(self.config, "retrieval_tau", 0.55)
        self.retrieval_alpha_max = getattr(self.config, "retrieval_alpha_max", 0.02)
        self.retrieval_beta_hidden = getattr(self.config, "retrieval_beta_hidden", 32)
        self.retrieval_beta_max = getattr(self.config, "retrieval_beta_max", 0.20)
        self.retrieval_beta_reg = getattr(self.config, "retrieval_beta_reg", 1e-4)

        self.register_buffer(
            "retrieval_gate_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        # ---------------- embedding ----------------
        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.dec_embedding = DataEmbedding(c_in=dec_in, d_model=d_model, dropout=self.dropout)

        # ---------------- encoder / decoder ----------------
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=e_layers,
            batch_first=True,
            dropout=self.dropout if e_layers > 1 else 0.0
        )

        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=d_layers,
            batch_first=True,
            dropout=self.dropout if d_layers > 1 else 0.0
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)

        # ---------------- router ----------------
        self.router = SampleRouterFromX(
            c_in=c_in,
            num_experts=self.num_experts,
            hidden=getattr(self.config, "router_hidden", 32),
            dropout=self.dropout
        )

        # ---------------- MoE Head ----------------
        self.moe_head = StandardMoEHead(
            d_model=d_model,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=min(self.dropout, 0.1),
        )

        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        # ---------------- retrieval gate ----------------
        self.beta_gate = RetrievalBetaGate(
            hidden_dim=self.retrieval_beta_hidden,
            beta_min=0.0,
            beta_max=self.retrieval_beta_max,
        )

        self.latest_aux_dict = {}

    # =========================================================
    # 5.1 backbone / gate 切换
    # =========================================================
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

    # =========================================================
    # 5.2 router balance loss
    # =========================================================
    def compute_sample_level_balance_loss(self, router_logits):
        router_prob = torch.softmax(router_logits, dim=-1)   # [B, E]
        load = router_prob.mean(dim=0)                       # [E]

        min_load = 0.2
        max_load = 0.5

        low_penalty = F.relu(min_load - load).pow(2)
        high_penalty = F.relu(load - max_load).pow(2)

        balance_loss = (low_penalty + high_penalty).sum()
        aux_dict = {
            "balance_loss": balance_loss.detach(),
            "expert_load": load.detach(),
        }
        return balance_loss, aux_dict

    # =========================================================
    # 5.3 retrieval memory index
    # =========================================================
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

        raise ValueError(f"Unsupported query shape: {queries.shape}")

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        bs = x.shape[0]
        if self.index == 0:
            raise RuntimeError("Retrieval index has not been constructed yet.")

        k = min(self.retrieval_num, self.index)

        queries = x
        keys = self.keys[:self.index]
        values = self.values[:self.index]

        dis = self.cosine_similarity(queries, keys)  # [B, N]

        if self.training and index is not None:
            self_range = torch.arange(
                -self.seq_len, self.seq_len + 1, device=x.device
            ).unsqueeze(0)

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

    # =========================================================
    # 5.4 backbone forward
    # =========================================================
    def _forward_backbone(
        self,
        x: torch.Tensor,
        dec_input: Optional[torch.Tensor] = None,
    ):
        head_router_logits = self.router(x)
        head_router_prob = torch.softmax(head_router_logits, dim=-1)

        head_topk_probs, head_topk_experts = torch.topk(
            head_router_prob, k=self.top_k_experts, dim=-1
        )

        head_mix_weights = head_topk_probs / (
            head_topk_probs.sum(dim=-1, keepdim=True) + 1e-8
        )

        x_emb_hist = self.enc_embedding(x)
        dec_emb = self.dec_embedding(dec_input)

        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)
        dec_out, _ = self.decoder(dec_emb, (h_n, c_n))
        dec_out = dec_out[:, -self.pred_len:, :]

        ctx, _ = self.xattn(dec_out, enc_out, enc_out)
        fused = torch.cat([dec_out, ctx], dim=-1)
        fused = self.fuse_proj(fused)
        final_shared = self.post_norm(fused)

        y = self.moe_head(
            final_shared,
            head_mix_weights,
            head_topk_experts
        )

        return y, head_router_logits, head_router_prob, head_topk_experts

    # =========================================================
    # 5.5 heuristic retrieval fusion
    # =========================================================
    def _heuristic_fuse(self, y, retrieval_pred, sims):
        sim_mean = sims.mean(dim=-1)
        dynamic_alpha = (sim_mean - self.retrieval_tau) / (1.0 - self.retrieval_tau + 1e-8)
        dynamic_alpha = dynamic_alpha.clamp(0.0, 1.0)
        dynamic_alpha = self.retrieval_alpha_max * dynamic_alpha
        dynamic_alpha = dynamic_alpha.view(-1, 1, 1)
        return (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_pred, dynamic_alpha

    # =========================================================
    # 5.6 learned gate fusion
    # =========================================================
    def _gate_fuse(self, x, y, sample_ids):
        retrieval_results, sims, _ = self.retrieval(x, sample_ids)
        retrieval_pred = retrieval_results[:, :, :self.out_dim]

        beta = self.beta_gate(
            x_enc=x,
            base_pred=y.detach(),
            ret_pred=retrieval_pred.detach(),
            sims=sims,
        )

        fused_y = (1.0 - beta) * y + beta * retrieval_pred
        return fused_y, retrieval_pred, sims, beta

    # =========================================================
    # 5.7 forward
    # =========================================================
    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        route_labels: Optional[torch.Tensor] = None,
        mode: str = "train",
        return_aux: bool = False,
    ):
        if mode == "gate_train":
            with torch.no_grad():
                y, head_router_logits, head_router_prob, head_topk_experts = self._forward_backbone(
                    x=x,
                    dec_input=dec_input,
                )
        else:
            y, head_router_logits, head_router_prob, head_topk_experts = self._forward_backbone(
                x=x,
                dec_input=dec_input,
            )

        aux_dict = {}
        total_aux_loss = y.new_tensor(0.0)

        if mode in {"gate_train", "gate_valid"}:
            y, retrieval_pred, sims, beta = self._gate_fuse(x, y, sample_ids)
            total_aux_loss = self.retrieval_beta_reg * beta.mean()

            aux_dict["beta_mean"] = beta.mean().detach()
            aux_dict["beta_max"] = beta.max().detach()
            aux_dict["sim_mean"] = sims.mean().detach()

        elif mode == "test" and hasattr(self, "index") and self.index > 0:
            if bool(self.retrieval_gate_ready.item()):
                y, retrieval_pred, sims, beta = self._gate_fuse(x, y, sample_ids)

                aux_dict["beta_mean"] = beta.mean().detach()
                aux_dict["beta_max"] = beta.max().detach()
                aux_dict["sim_mean"] = sims.mean().detach()
            else:
                retrieval_results, sims, _ = self.retrieval(x, sample_ids)
                retrieval_pred = retrieval_results[:, :, :self.out_dim]
                y, dynamic_alpha = self._heuristic_fuse(y, retrieval_pred, sims)

                aux_dict["beta_mean"] = dynamic_alpha.mean().detach()
                aux_dict["beta_max"] = dynamic_alpha.max().detach()
                aux_dict["sim_mean"] = sims.mean().detach()

        if mode in {"train", "valid"}:
            balance_loss, balance_aux_dict = self.compute_sample_level_balance_loss(
                router_logits=head_router_logits
            )
            total_aux_loss = total_aux_loss + 0.1 * balance_loss
            aux_dict.update(balance_aux_dict)

        aux_dict["total_aux_loss"] = total_aux_loss.detach()
        aux_dict["router_prob"] = head_router_prob.detach()
        aux_dict["topk_experts"] = head_topk_experts.detach()

        self.latest_aux_dict = aux_dict

        if return_aux:
            return y, total_aux_loss, aux_dict
        return y, total_aux_loss
