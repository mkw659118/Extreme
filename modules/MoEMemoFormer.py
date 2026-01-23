#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict, Tuple, List
from layers.embedding import DataEmbedding

class NormalHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, pred_len, d_model]
        return self.proj(x)


class MidHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.fc = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)   # 关键：保留 proj

    def forward(self, x):
        x = self.drop(self.act(self.fc(x)))
        return self.proj(x)


class ExtremeHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)     # 关键：压回 d_model
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)         # 关键：所有专家统一 proj: d_model -> 1

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return self.proj(x)

def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    if win_size is None or win_size <= 0 or win_size > seq_len:
        win_size = max(1, seq_len // 2)

    upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)

    # left-window clipping
    if win_size < seq_len:
        for i in range(seq_len):
            left = max(0, i - win_size + 1)
            upper[i, :left] = True

    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_bias.masked_fill_(upper, float("-inf"))
    return attn_bias


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=256, dropout=0.3):
        super().__init__()
        d_ff = d_ff or (d_model * 4)
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x1 = self.norm1(x)
        y = self.attn(x1, x1, x1, attn_mask=attn_mask)[0]
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x

# class SampleRouterFromX(nn.Module):
#     def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(c_in, hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, num_experts),
#         )

#     def forward(self, x):
#         """
#         x: [B, seq_len, c_in]
#         returns logits: [B, E]
#         """
#         feat = x.mean(dim=1)  # [B, c_in]
#         return self.net(feat)

class SampleRouterFromX(nn.Module):
   
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.c_in = c_in
        in_dim = 3 * c_in  # std + max_abs + last

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        # 1) std: [B, C]
        std = x.std(dim=1, unbiased=False)

        # 2) max_abs: [B, C]
        max_abs = x.abs().amax(dim=1)

        # 3) last: [B, C]
        last = x[:, -1, :]

        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)
    

class ThreeExpertPatchTransformer(nn.Module):

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        d_model: int,
        win_size: int,
        revin: bool,
        num_heads: int,
        use_memory: bool,
        num_layers_intra_patch: int,
        num_layers_inter_patch: int,
        config=None,
        c_in: int = 10,
    ):
        super().__init__()
        self.config = config
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.patch_len = patch_len
        self.win_size = win_size

        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len

        # -------- expert definition --------
        self.gmm_start = int(getattr(config, "gmm_start", 2))
        self.gmm_end   = int(getattr(config, "gmm_end", 5))
        assert 0 <= self.gmm_start < self.gmm_end <= c_in, "Invalid gmm_start/gmm_end"
        self.num_experts = int(self.gmm_end - self.gmm_start)

        self.teacher_forcing = True

        # -------- Embedding + pred tokens --------
        dropout = float(getattr(config, "dropout", 0.3))
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        # -------- backbone --------
        d_ff = int(getattr(config, "d_ff", 256))
        self.intra = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_intra_patch)
        ])
        self.inter = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_inter_patch)
        ])
        self.post_norm = nn.RMSNorm(d_model)

        # -------- router --------
        router_hidden = int(getattr(config, "router_x_hidden", self.d_model))
        router_dropout = float(getattr(config, "router_x_dropout", 0.3))
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # # -------- heads --------
        # self.expert_heads = nn.ModuleList([ExpertHead(d_model) for _ in range(self.num_experts)])
        assert self.num_experts == 3, "此处示例按3专家写的"
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=dropout),
            ExtremeHead(d_model, hidden=2*d_model, dropout=dropout),
        ])
        
        # label2expert[k] = GMM标签k 应该路由到的 expert id
        # 例如：如果 GMM label=2 是 normal，label=0 是 mid，label=1 是 extreme
        # 那么 label2expert = [mid, extreme, normal] = [1, 2, 0]
        label2expert = getattr(config, "label2expert", None)
        if label2expert is None:
            label2expert = list(range(self.num_experts))
        self.register_buffer("label2expert", torch.tensor(label2expert, dtype=torch.long))
        
                # -------- top-k gating --------
        self.top_k = int(getattr(config, "top_k", 2))          # 你要 top2
        self.router_tau = float(getattr(config, "router_tau", 1.0))  # softmax 温度
        self.tf_blend = float(getattr(config, "tf_blend", 0.0))      # 0=不使用GMM强制混合(避免“硬编码”)

        # -------- GMM label slices--------
        self.gmm_pt_start  = 2
        self.gmm_pt_end    = 5
        self.gmm_seq_start = 7
        self.gmm_seq_end   = 10

    def _gmm_argmax_label(self, x: torch.Tensor) -> torch.Tensor:
        pt  = x[:, :, self.gmm_pt_start:self.gmm_pt_end]
        seq = x[:, :, self.gmm_seq_start:self.gmm_seq_end]

        if pt.size(-1) != seq.size(-1):
            gmm = x[:, :, self.gmm_start:self.gmm_end]
        else:
            gmm = seq + 0.4 * pt

        gmm_mean = gmm.mean(dim=1)
        label = torch.argmax(gmm_mean, dim=-1)
        return label

    def forward_backbone(self, x_emb: torch.Tensor, intra_mask, inter_mask):
        B = x_emb.size(0)

        patches = rearrange(x_emb, "b (p pl) d -> b p pl d", p=self.num_patches, pl=self.patch_len)

        patches_intra = rearrange(patches, "b p pl d -> (b p) pl d").contiguous()
        for blk in self.intra:
            patches_intra = blk(patches_intra, attn_mask=intra_mask)
        patches_intra = rearrange(patches_intra, "(b p) pl d -> b p pl d", b=B, p=self.num_patches).contiguous()

        intra_tokens = rearrange(patches_intra, "b p pl d -> b (p pl) d")

        inter_patches = rearrange(patches_intra, "b p pl d -> (b pl) p d").contiguous()
        for blk in self.inter:
            inter_patches = blk(inter_patches, attn_mask=inter_mask)
        inter_tokens = rearrange(inter_patches, "(b pl) p d -> b (p pl) d", b=B, pl=self.patch_len)

        return self.post_norm(intra_tokens + inter_tokens)


    # def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
    #     """
    #     x: [B, seq_len, c_in]
    #     y_true (optional): [B, pred_len, D] or [B, pred_len, 1]
    #     """
    #     B = x.size(0)

    #     # ---------------- routing ----------------
    #     router_logits = self.router(x)                  # [B, E]

    #     if route_labels is None:
    #         route_labels = self._gmm_argmax_label(x)  # [B]
        
    #     if self.training and self.teacher_forcing:
    #         expert_idx = route_labels
    #     else:
    #         expert_idx = torch.argmax(router_logits, dim=-1)

    #     # ---------------- embedding + pred tokens ----------------
    #     x_emb_hist = self.embedding(x)                               # [B, seq_len, d_model]
    #     pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)      # [B, pred_len, d_model]
    #     x_emb = torch.cat([x_emb_hist, pred_token], dim=1)                # [B, total_len, d_model]

    #     intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
    #     inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

    #     # ---------------- shared backbone ----------------
    #     final_shared = self.forward_backbone(x_emb, intra_mask, inter_mask)   # [B, total_len, d_model]
    #     final_shared = final_shared[:, -self.pred_len:, :]                             # [B, pred_len, d_model]

    #     # ---------------- heads ----------------
    #     y_all = torch.cat([h(final_shared) for h in self.expert_heads], dim=-1)         # [B, pred_len, E]
    #     idx = expert_idx.view(B, 1, 1).expand(B, self.pred_len, 1)
    #     y = y_all.gather(dim=-1, index=idx)                                # [B, pred_len, 1]

    #     return y
    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        y_true (optional): [B, pred_len, D] or [B, pred_len, 1]
        return: y [B, pred_len, 1]
        """
        B = x.size(0)

        # ---------------- routing ----------------
        router_logits = self.router(x)  # [B, E]

        # route_labels: 若外部没传，则用GMM argmax，并映射到 expert id（否则 label2expert 永远没用到）
        if route_labels is None:
            gmm_label = self._gmm_argmax_label(x)                # [B] in [0..E-1]
            route_labels = self.label2expert[gmm_label]          # [B] expert id
        else:
            # 假设外部传入的 route_labels 已经是 expert id
            route_labels = route_labels.long()

        # router 概率（带温度）
        tau = max(self.router_tau, 1e-6)
        router_prob = torch.softmax(router_logits / tau, dim=-1)  # [B, E]

        # 可选：训练时把“GMM路由标签”作为 soft prior 混入概率（不是硬选某个专家）
        # tf_blend=0 代表完全不混入（最“去硬编码”）
        if self.training and self.teacher_forcing and self.tf_blend > 0.0:
            onehot = F.one_hot(route_labels, num_classes=self.num_experts).to(router_prob.dtype)  # [B, E]
            router_prob = (1.0 - self.tf_blend) * router_prob + self.tf_blend * onehot

        # ---------------- embedding + pred tokens ----------------
        x_emb_hist = self.embedding(x)                                    # [B, seq_len, d_model]
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)      # [B, pred_len, d_model]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)                # [B, total_len, d_model]

        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # ---------------- shared backbone ----------------
        final_shared = self.forward_backbone(x_emb, intra_mask, inter_mask)      # [B, total_len, d_model]
        final_shared = final_shared[:, -self.pred_len:, :]                       # [B, pred_len, d_model]

        # ---------------- heads: compute all experts ----------------
        # y_all: [B, pred_len, E]
        y_all = torch.cat([h(final_shared) for h in self.expert_heads], dim=-1)

        # ---------------- top-k mix (top2) ----------------
        k = min(self.top_k, self.num_experts)
        topk = torch.topk(router_prob, k=k, dim=-1)      # values/indices: [B, k]
        topk_w = topk.values                              # [B, k]
        topk_idx = topk.indices                           # [B, k]

        # 归一化 topk 权重（保证两项权重和为1）
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

        # 从 y_all 取出 topk 专家对应的输出: [B, pred_len, k]
        gather_idx = topk_idx.view(B, 1, k).expand(B, self.pred_len, k)
        y_topk = y_all.gather(dim=-1, index=gather_idx)  # [B, pred_len, k]

        # 加权求和得到最终输出: [B, pred_len, 1]
        w = topk_w.view(B, 1, k).expand(B, self.pred_len, k)
        y = (y_topk * w).sum(dim=-1, keepdim=True)

        return y