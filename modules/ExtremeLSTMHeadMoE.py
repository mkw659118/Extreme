from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding
from layers.att.cross_attention import CrossAttention


# =========================================================
# 1. 基础专家：输入/输出维度都保持 d_model
# =========================================================
class MoEExpert(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)

        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)

        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.fc3(x))
        return x


# =========================================================
# 2. Router：只基于输入样本统计量进行路由
#    不直接吃 route_label
# =========================================================
class SampleRouterFromX(nn.Module):
    """
    输入:
        x: [B, L, C]
    输出:
        logits: [B, num_experts]
    """
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.c_in = c_in

        # 三类样本统计特征：std + max_abs + last
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
# 3. 标准 MoE Head
#    - expert 输出 d_model
#    - 最后共享线性层映射到 out_dim
#    - 不直接使用 route_label 修改输出
# =========================================================
class StandardMoEHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int = 1,
        num_experts: int = 2,
        top_k: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            MoEExpert(d_model=d_model, hidden=2 * d_model, dropout=dropout)
            for _ in range(num_experts)
        ])
        
        # self.moe_norm = nn.RMSNorm(d_model)

        # 所有专家共享的最终输出投影
        self.final_proj = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x: torch.Tensor,             # [B, pred_len, d_model]
        router_probs: torch.Tensor,  # [B, K]，已归一化后的 top-k 混合权重
        topk_experts: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        B, pred_len, D = x.shape

        # 聚合后的专家输出特征
        moe_out = torch.zeros_like(x)  # [B, pred_len, d_model]

        for k in range(self.top_k):
            expert_idx = topk_experts[:, k]             # [B]
            weight = router_probs[:, k].view(B, 1, 1)  # [B, 1, 1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if not mask.any():
                    continue

                # 当前 expert 处理属于自己的样本
                expert_feat = self.experts[e](x[mask])         # [mask_B, pred_len, d_model]
                moe_out[mask] += expert_feat * weight[mask]    # 加权融合
        # moe_out = self.moe_norm(moe_out)
        moe_out = x + moe_out
        final_out = self.final_proj(moe_out)  # [B, pred_len, out_dim]
        return final_out


# =========================================================
# 4. 主模型：ExtremeLSTMMemo
#    - LSTM encoder-decoder
#    - CrossAttention
#    - MoE head
#    - Retrieval
#    - route_label 软监督
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
        dec_in: int = 3,      # !!! 你新版数据集 label 现在是 3 列，所以这里建议设为 3
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
        self.retrieval_num = 4
        self.top_k_experts = 2
        self.retrieval_stride = 1

        # ---------------- loss weights ----------------
        self.importance_loss_weight = 0.1
        self.load_loss_weight = 0.1

        # route supervision 的权重
        self.route_loss_weight = getattr(self.config, "route_loss_weight", 0.01)
        self.use_route_supervision = getattr(self.config, "use_route_supervision", False)

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
            hidden=d_model,
            dropout=self.dropout
        )

        # ---------------- MoE Head ----------------
        self.moe_head = StandardMoEHead(
            d_model=d_model,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=self.dropout
        )

        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        # 保存最近一次 forward 的辅助信息，方便训练打印
        self.latest_aux_dict = {}

    # =========================================================
    # 4.1 router balance loss
    # =========================================================
    def compute_switch_balance_loss(
        self,
        router_logits: torch.Tensor,   # [B, E]
    ) -> torch.Tensor:
        """
        Switch Transformer 风格的辅助路由均衡损失
        适用于 top-1 routing
        """
        router_prob = torch.softmax(router_logits, dim=-1)   # [B, E]
        num_experts = router_prob.size(-1)

        # P_i: soft importance
        prob_per_expert = router_prob.mean(dim=0)            # [E]

        # f_i: hard load
        top1_expert = torch.argmax(router_prob, dim=-1)      # [B]
        one_hot = F.one_hot(top1_expert, num_classes=num_experts).float()
        frac_per_expert = one_hot.mean(dim=0)                # [E]

        # Switch auxiliary loss
        aux_loss = num_experts * torch.sum(prob_per_expert * frac_per_expert)

        return aux_loss

    def compute_balance_loss(
        self,
        router_prob: torch.Tensor
    ):
        balance_loss = self.compute_switch_balance_loss(router_prob)

        aux_dict = {
            "balance_loss": balance_loss.detach()
        }
        return balance_loss, aux_dict

    # =========================================================
    # 4.2 route supervision loss
    #     route_label 不是硬绑定，而是 router 的软监督
    # =========================================================
    def compute_route_supervision_loss(
        self,
        router_logits: torch.Tensor,                 # [B, E]
        route_labels: Optional[torch.Tensor] = None  # [B]
    ) -> torch.Tensor:
        """
        route_label 对 router 的软监督损失
        这里只在训练期使用
        """
        if route_labels is None:
            return torch.tensor(0.0, device=router_logits.device)

        valid_mask = route_labels >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=router_logits.device)

        valid_logits = router_logits[valid_mask]
        valid_labels = route_labels[valid_mask].long()

        # 注意：这里默认 route_label 的类别数 <= num_experts
        route_loss = F.cross_entropy(valid_logits, valid_labels)
        return route_loss

    # =========================================================
    # 4.3 retrieval memory index
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
        """
        queries: [B, L, C] or [B, D]
        keys   : [N, L, C] or [N, D]
        return : [B, N]
        """
        if len(queries.shape) == 3:
            B = queries.size(0)
            N = keys.size(0)
            queries = queries.reshape(B, -1)
            keys = keys.reshape(N, -1)

            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())

        elif len(queries.shape) == 2:
            q_norm = F.normalize(queries, p=2, dim=-1)
            k_norm = F.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())

        else:
            raise ValueError(f"Unsupported query shape: {queries.shape}")

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        bs = x.shape[0]
        k = min(self.retrieval_num, self.index)

        queries = x
        keys = self.keys[:self.index]
        values = self.values[:self.index]

        dis = self.cosine_similarity(queries, keys)  # [B, N]

        # 训练时屏蔽自身附近样本
        if self.training and index is not None:
            self_range = torch.arange(
                -self.seq_len, self.seq_len + 1, device=x.device
            ).unsqueeze(0)  # [1, 2*seq_len+1]

            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[invalid_index < 0] = 0
            invalid_index[invalid_index >= self.index] = self.index - 1

            row_idx = torch.arange(bs, device=x.device).unsqueeze(1).repeat(1, invalid_index.size(1))
            dis[row_idx, invalid_index] = -100.0

        dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)                        # [B, k]
        sims = dis_topk
        print(sims)
        probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)     # [B, k, 1, 1]

        retrieved_values = values[indices_topk]                                      # [B, k, pred_len, dec_in]
        output = torch.sum(probs_topk * retrieved_values, dim=1)                     # [B, pred_len, dec_in]

        return output, sims, 0

    # =========================================================
    # 4.4 forward
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
        # ---------------- router ----------------
        head_router_logits = self.router(x)                    # [B, E]
        head_router_prob = torch.softmax(head_router_logits, dim=-1)

        head_topk_probs, head_topk_experts = torch.topk(
            head_router_prob, k=self.top_k_experts, dim=-1
        )                                                      # [B, K], [B, K]

        # top-k 概率重新归一化，得到专家混合权重
        head_mix_weights = head_topk_probs / (
            head_topk_probs.sum(dim=-1, keepdim=True) + 1e-8
        )                                                      # [B, K]

        # ---------------- embedding ----------------
        x_emb_hist = self.enc_embedding(x)                     # [B, seq_len, d_model]
        dec_emb = self.dec_embedding(dec_input)                # [B, label_len+pred_len, d_model] or [B, pred_len, d_model]

        # ---------------- encoder / decoder ----------------
        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)         # [B, seq_len, d_model]
        dec_out, _ = self.decoder(dec_emb, (h_n, c_n))         # [B, dec_len, d_model]
        dec_out = dec_out[:, -self.pred_len:, :]               # 只取最后 pred_len 步

        # ---------------- cross attention + fuse ----------------
        ctx, _ = self.xattn(dec_out, enc_out, enc_out)         # [B, pred_len, d_model]
        fused = torch.cat([dec_out, ctx], dim=-1)              # [B, pred_len, 2*d_model]
        fused = self.fuse_proj(fused)                          # [B, pred_len, d_model]
        final_shared = self.post_norm(fused)                   # [B, pred_len, d_model]

        # ---------------- MoE Head ----------------
        y = self.moe_head(
            final_shared,
            head_mix_weights,
            head_topk_experts
        )                                                      # [B, pred_len, out_dim]

        # ---------------- retrieval (test only) ----------------
        # if mode == "test":
        #     retrieval_results, sims, _ = self.retrieval(x, sample_ids)

        #     # retrieval_results shape: [B, pred_len, dec_in]
        #     # 这里只取第 0 维作为预测目标融合（因为 out_dim=1）
        #     retrieval_pred = retrieval_results[:, :, :self.out_dim]

        #     # sim_mean = torch.mean(sims, dim=-1, keepdim=True)  # [B, 1]
        #     # dynamic_alpha = 0.01* (
        #     #     (sim_mean - sim_mean.min()) /
        #     #     (sim_mean.max() - sim_mean.min() + 1e-8)
        #     # )
        #     # dynamic_alpha = dynamic_alpha.unsqueeze(-1)        # [B, 1, 1]
        #     sim_mean = torch.mean(sims, dim=-1, keepdim=True)   # [B, 1]
        #     batch_sim_mean = sim_mean.mean().item()             # 标量

        #     if batch_sim_mean > 0.95:
        #         dynamic_alpha = 0.3
        #     else:
        #         dynamic_alpha = 0.05
            

        #     y = (1 - dynamic_alpha) * y + dynamic_alpha * retrieval_pred

        # ---------------- balance loss ----------------
        balance_loss, aux_dict = self.compute_balance_loss(
            router_prob=head_router_prob
        )

        # ---------------- route supervision loss ----------------
        if self.use_route_supervision and (mode == "train"):
            route_loss = self.compute_route_supervision_loss(
                router_logits=head_router_logits,
                route_labels=route_labels
            )
        else:
            route_loss = torch.tensor(0.0, device=x.device)

        # total_aux_loss = 0.1* balance_loss + self.route_loss_weight * route_loss
        total_aux_loss = 0.1* balance_loss
        # total_aux_loss = self.route_loss_weight * route_loss

        aux_dict["route_loss"] = route_loss.detach()
        aux_dict["total_aux_loss"] = total_aux_loss.detach()
        aux_dict["router_prob"] = head_router_prob.detach()
        aux_dict["topk_experts"] = head_topk_experts.detach()

        self.latest_aux_dict = aux_dict

        if return_aux:
            return y, total_aux_loss, aux_dict
        return y, total_aux_loss
        # return y, torch.tensor(0.0, device=total_aux_loss.device, requires_grad=True)
    