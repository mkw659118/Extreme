# #Author  :   mkw 
# #Time    :   2025/09/12 19:31:08
# #Desc    :   None

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from layers.embedding import DataEmbedding


# def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
#     """
#     生成因果窗口掩码。用于限制每个位置仅能与前面的窗口大小范围内的元素进行注意力计算。
#     """
#     bad = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
#     for i in range(seq_len):
#         left = max(0, i - win_size + 1)
#         bad[i, :left] = True
#     attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
#     attn_bias.masked_fill_(bad, torch.finfo(attn_bias.dtype).min)
#     return attn_bias


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
#         super().__init__()
#         d_ff = d_ff or (d_model * 4)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model), nn.Dropout(dropout)
#         )

#     def forward(self, x, attn_mask=None):  # x: [B, L, d]
#         y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)[0]
#         x = x + y
#         x = x + self.ff(self.norm2(x))
#         return x

# # =========================
# # Bucketed Extreme Memory (per patch position)
# # =========================
# class BucketedExtremeMemory(nn.Module):
#     """
#     每个 patch 位置一个桶：
#     - keys[p]: [M, d]
#     - vals[p]: [M, d]  (这里存“原型向量”，用作特征级修正；更稳)
#       若你更偏好“存残差(标量/短向量)”，可以把 vals 调整为 [M, 1] 并改融合方式即可
#     写入：仅极端 patch；读取：Top-K 余弦，softmax 融合
#     """
#     def __init__(self, num_buckets: int, d_model: int, mem_size_per_bucket: int = 256,
#                  topk: int = 16, temperature: float = 0.5, ema_momentum: float = 0.2, device=None):
#         super().__init__()
#         self.P = num_buckets
#         self.d = d_model
#         self.M = mem_size_per_bucket
#         self.K = topk
#         self.tau = temperature
#         self.momentum = ema_momentum
#         self.device = device

#         # 初始化为单位化高斯
#         self.keys = nn.ParameterList([
#             nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
#             for _ in range(self.P)
#         ])
#         self.vals = nn.ParameterList([
#             nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
#             for _ in range(self.P)
#         ])
#         # 环形指针（缓写可改为原型 EMA；这里两者都支持）
#         self.register_buffer('ptr', torch.zeros(self.P, dtype=torch.long))

#         # 是否使用“仅EMA原型”模式（True：不使用逐条写入，直接 EMA 到所有槽位；False：环形写入）
#         self.ema_only = True

#     @torch.no_grad()
#     def write(self, p: int, k_batch: torch.Tensor, v_batch: torch.Tensor, w_batch: torch.Tensor = None):
#         """
#         p: 桶索引
#         k_batch: [B_p, d]
#         v_batch: [B_p, d]  (通常用查询 patch 的上下文 token 或其投影)
#         w_batch: [B_p, K] 或 [B_p, 1] 权重；可选
#         """
#         if k_batch.numel() == 0:
#             return

#         k_batch = F.normalize(k_batch, dim=-1)
#         v_batch = F.normalize(v_batch, dim=-1)

#         if self.ema_only:
#             # 用平均或加权平均作为“更新方向”，向所有槽位做 EMA（近似原型更新）
#             if w_batch is None:
#                 direction = k_batch.mean(dim=0, keepdim=True)  # [1, d]
#                 value_dir = v_batch.mean(dim=0, keepdim=True)  # [1, d]
#             else:
#                 # 若有权重，按权求平均
#                 ws = w_batch
#                 if ws.dim() == 1:
#                     ws = ws.unsqueeze(-1)
#                 ws = ws / (ws.sum(dim=0, keepdim=True) + 1e-8)
#                 direction = (ws.t() @ k_batch)  # [*, d]
#                 value_dir = (ws.t() @ v_batch)
#                 direction = direction.mean(dim=0, keepdim=True)
#                 value_dir = value_dir.mean(dim=0, keepdim=True)

#             new_k = F.normalize((1 - self.momentum) * self.keys[p].data + self.momentum * direction, dim=-1)
#             new_v = F.normalize((1 - self.momentum) * self.vals[p].data + self.momentum * value_dir, dim=-1)
#             self.keys[p].data = new_k
#             self.vals[p].data = new_v
#         else:
#             # 逐条环形写入（若希望严格保存具体样本）
#             b = k_batch.size(0)
#             start = int(self.ptr[p].item())
#             idx = (torch.arange(b, device=k_batch.device) + start) % self.M
#             self.keys[p].data[idx] = k_batch
#             self.vals[p].data[idx] = v_batch
#             self.ptr[p] = (self.ptr[p] + b) % self.M

#     def read(self, p: int, q: torch.Tensor, topk=None):
#         """
#         q: [B, d] 查询
#         return:
#           m:  [B, d]  记忆融合后的向量
#           s:  [B, 1]  最大相似度（门控参考）
#           wK: [B, M]  （可选）稀疏权重（仅TopK处非零）
#         """
#         if topk is None:
#             topk = self.K
#         Kmat = F.normalize(self.keys[p], dim=-1)                 # [M, d]
#         qn = F.normalize(q, dim=-1)                              # [B, d]
#         sim = (qn @ Kmat.t()) / self.tau                         # [B, M]
#         k = min(topk, self.M)
#         topv, topi = torch.topk(sim, k=k, dim=-1)                # [B, k], [B, k]
#         w = torch.softmax(topv, dim=-1)                          # [B, k]
#         Vmat = F.normalize(self.vals[p], dim=-1)                 # [M, d]
#         picked = Vmat[topi]                                      # [B, k, d]
#         m = (w.unsqueeze(-1) * picked).sum(dim=1)                # [B, d]
#         s, _ = sim.max(dim=-1, keepdim=True)                     # [B, 1]

#         # 构造稀疏权重（可用于可视化/调试）
#         wK = torch.zeros_like(sim)
#         wK.scatter_(-1, topi, w)
#         return m, s, wK


# class PatchExtremeMemoryTransformer(nn.Module):
#     """
#     输入:  x ∈ [B, seq_len, 1]
#     输出:  y ∈ [B, pred_len, 1]
#     结构:
#       - 窗口级样本 -> 切 P 个 patch
#       - patch内 Transformer 堆叠
#       - patch间 Transformer 堆叠（token = 每个 patch 的平均 token）
#       - 分桶记忆: 每个位置一个桶, 只在极端 patch 写入, 推理仅读
#       - 门控融合: g = σ(MLP([token, sim_max]))；out = out + g * fuse(out, mem)
#       - 反patchify回 token, 线性投影 -> 输出
#     """
#     def __init__(
#         self,
#         seq_len: int,
#         pred_len: int,
#         patch_len: int,
#         d_model: int,
#         revin: bool,
#         num_heads: int,
#         num_layers_in_patch: int = 2,
#         num_layers_inter_patch: int = 1,
#         d_ff: int = None,
#         dropout: float = 0.1,
#         mem_size_per_bucket: int = 256,
#         mem_topk: int = 16,
#         mem_tau: float = 0.5,
#         mem_momentum: float = 0.2,
#         z_thresh: float = 2.0,
#         diff_z_thresh: float = 2.0,
#         use_diff: bool = True,
#         warmup_mem_steps: int = 200,
#         warmup_ratio: float = 0.10,
#         force_top1_steps: int = 50,
#     ):
#         super().__init__()
#         self.revin = revin
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.total_len = seq_len + pred_len
#         self.patch_len = patch_len
#         assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
#         self.num_patches = self.total_len // self.patch_len

#         # Embedding + 预投影到 total_len（让后部patch也有token）
#         self.embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=dropout)
#         self.predict_linear = nn.Linear(seq_len, self.total_len)  # [B, d, seq] -> [B, d, total]

#         # patch 内 Transformer 堆叠（每个patch一个堆栈）
#         self.in_patch_blocks = nn.ModuleList([
#             nn.ModuleList([
#                 TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
#                 for _ in range(num_layers_in_patch)
#             ]) for _ in range(self.num_patches)
#         ])

#         # patch 间 Transformer 堆叠（token为每patch的平均token）
#         self.inter_patch_blocks = nn.ModuleList([
#             TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
#             for _ in range(num_layers_inter_patch)
#         ])

#         self.post_norm = nn.LayerNorm(d_model)
#         self.proj_out = nn.Linear(d_model, 1)

#         # 记忆（按位置分桶）
#         self.use_memory = True
#         self.memory = BucketedExtremeMemory(
#             num_buckets=self.num_patches, d_model=d_model,
#             mem_size_per_bucket=mem_size_per_bucket, topk=mem_topk,
#             temperature=mem_tau, ema_momentum=mem_momentum
#         )
#         self.mem_fuse = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.GELU(),
#             nn.Linear(d_model, d_model)
#         )
#         self.gate_proj = nn.Sequential(
#             nn.Linear(d_model + 1, d_model),
#             nn.Sigmoid()
#         )
#         self.mem_scale = nn.Parameter(torch.tensor(1.0))
#         self.gate_bias = nn.Parameter(torch.zeros(1))

#         # 极端判定 & 训练暖身
#         self.z_thresh = z_thresh
#         self.diff_z_thresh = diff_z_thresh
#         self.use_diff = use_diff
#         self.ext_percentile = 0.9
#         self.warmup_mem_steps = warmup_mem_steps
#         self.warmup_ratio = warmup_ratio
#         self.force_top1_steps = force_top1_steps
#         self._global_step = 0

#     # ---------- masks ----------
#     def _build_masks(self, device, dtype=torch.float32):
     
#         # 生成 patch 内部掩码，形状为 [patch_len, patch_len]
#         intra_patch_mask = generate_causal_window_mask(self.patch_len, self.patch_len, device, dtype)
        
#         # 生成 patch 间掩码，形状为 [num_patches, num_patches]
#         inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, device, dtype)

#         # 扩展 inter_mask 的形状，以适配总长度
#         inter_mask_full = torch.zeros(self.total_len, self.total_len, dtype=dtype, device=device)
#         inter_mask_full[:self.num_patches, :self.num_patches] = inter_mask

#         return intra_patch_mask, inter_mask_full

#     # ---------- 极端 patch 掩码 ----------
#     def _patch_extreme_mask(self, x_raw, means, stdev):
#         """
#         基于 z-score 和/或差分 z-score 的极端强度，按 patch 汇聚到 [B, P] 掩码
#         """
#         B, L_in, C = x_raw.shape
#         T = self.total_len
#         device = x_raw.device

#         z = (x_raw - means) / (stdev + 1e-5)
#         z_abs = z.abs().amax(dim=-1)  # [B, L_in]

#         if L_in < T:
#             pad = torch.zeros(B, T - L_in, device=device, dtype=z_abs.dtype)
#             z_abs_full = torch.cat([z_abs, pad], dim=1)
#         else:
#             z_abs_full = z_abs[:, :T]

#         z_abs_patch = rearrange(z_abs_full, 'b (np pl) -> b np pl',
#                                 np=self.num_patches, pl=self.patch_len)
#         z_per_patch = z_abs_patch.amax(dim=-1)  # [B, P]
#         z_flag = (z_per_patch > self.z_thresh)

#         d_flag = torch.zeros_like(z_flag)
#         if self.use_diff:
#             if L_in >= 2:
#                 diff = x_raw[:, 1:, :] - x_raw[:, :-1, :]
#                 diff_z = diff / (stdev + 1e-5)
#                 diff_abs = diff_z.abs().amax(dim=-1)  # [B, L_in-1]
#                 diff_abs = torch.cat([torch.zeros(B, 1, device=device, dtype=diff_abs.dtype), diff_abs], dim=1)
#             else:
#                 diff_abs = torch.zeros(B, L_in, device=device, dtype=z_abs.dtype)

#             if L_in < T:
#                 pad2 = torch.zeros(B, T - L_in, device=device, dtype=diff_abs.dtype)
#                 diff_full = torch.cat([diff_abs, pad2], dim=1)
#             else:
#                 diff_full = diff_abs[:, :T]

#             diff_patch = rearrange(diff_full, 'b (np pl) -> b np pl',
#                                    np=self.num_patches, pl=self.patch_len)
#             d_per_patch = diff_patch.amax(dim=-1)
#             d_flag = (d_per_patch > self.diff_z_thresh)

#         flag = (z_flag | d_flag)

#         # 若都不触发，用分位增强（避免 warm-start 冷库）
#         if (~flag).all():
#             strength = z_per_patch
#             if self.use_diff:
#                 strength = torch.maximum(strength, d_per_patch)
#             q = torch.quantile(strength, self.ext_percentile, dim=1, keepdim=True)
#             flag = (strength >= q)

#         # 训练早期强制保留 top-k patch（只写不读阶段构库）
#         if (~flag).all() and self._global_step < self.warmup_mem_steps:
#             strength = z_per_patch
#             if self.use_diff:
#                 strength = torch.maximum(strength, d_per_patch)
#             k = max(1, int(self.num_patches * self.warmup_ratio))
#             topk = torch.topk(strength, k=k, dim=1).indices
#             flag = torch.zeros_like(strength, dtype=torch.bool)
#             flag.scatter_(1, topk, True)

#         if (~flag).all() and self._global_step < self.force_top1_steps:
#             strength = z_per_patch
#             if self.use_diff:
#                 strength = torch.maximum(strength, d_per_patch)
#             top1 = torch.topk(strength, k=1, dim=1).indices
#             flag = torch.zeros_like(strength, dtype=torch.bool)
#             flag.scatter_(1, top1, True)

#         return flag  # [B, P] bool
    

#     def forward(self, x, x_mark=None,):  # x: [B, seq_len, 1]
#         B, L, C = x.shape
#         assert L == self.seq_len and C == 1

#         x_raw = x  # 用于极端掩码评估

#         # ---------- RevIN ----------
#         if self.revin:
#             mean = x.mean(1, keepdim=True).detach()  # [B,1,1]
#             x_centered = x - mean
#             std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,1]
#             x = x_centered / std
#             stats = (mean, std)

#         else:
#             mean = x.mean(1, keepdim=True).detach()
#             std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
#             stats = (mean, std)

#         # ---------- Embedding ----------
#         x_emb = self.embedding(x)                          # [B, L, d]
#         x_emb = rearrange(x_emb, 'b l d -> b d l')         # [B, d, L]
#         x_emb = self.predict_linear(x_emb)                 # [B, d, total_len]
#         x_emb = rearrange(x_emb, 'b d l -> b l d')         # [B, total_len, d]

#         # ---------- Patchify ----------
#         patches = rearrange(x_emb, 'b (np pl) d -> b np pl d',np=self.num_patches, pl=self.patch_len)  # [B, P, pl, d]

#         # ---------- Masks (float: -inf/0，符合 MHA 要求) ----------
#         intra_patch_mask = generate_causal_window_mask(
#             seq_len=self.patch_len, win_size=self.patch_len,
#             device=patches.device, dtype=patches.dtype
#         )  # [pl, pl], float(-inf/0)
        
#         inter_patch_mask = generate_causal_window_mask(
#             seq_len=self.num_patches, win_size=self.num_patches,
#             device=patches.device, dtype=patches.dtype
#         )  # [P, P], float(-inf/0)

#         # ---------- Intra-patch attention（局部） ----------
#         patch_outs = []
#         for p in range(self.num_patches):
#             out = patches[:, p]  # [B, pl, d]
#             for block in self.in_patch_blocks[p]:
#                 out = block(out, attn_mask=intra_patch_mask)     # [B, pl, d]
#             # 不再额外 post_norm，避免与 TransformerBlock 内部的 norm 重复
#             patch_outs.append(out)
#         patches_local = torch.cat(patch_outs, dim=1)        # [B, P*pl, d]

#         # ---------- Patch-level patches ----------
#         patch_reps = patches_local.view(B, self.num_patches, self.patch_len, -1).mean(dim=2)  # [B, P, d]

#         # ---------- Extreme mask（用于选择写入/强融合的 patch） ----------
#         mean_for_mask, std_for_mask = stats
#         extreme_mask = self._patch_extreme_mask(x_raw, mean_for_mask, std_for_mask)  # [B, P] bool

#         # ---------- 分桶记忆 + 门控融合（在 patch token 上进行） ----------
#         if self.use_memory:
#             fused_list = []  # 用 cat 的方式拼回 [B,P,d]
#             for p in range(self.num_patches):
#                 q_pool = patch_reps[:, p, :]                               # [B, d]
#                 # 读
#                 m_read, sim_max, _ = self.memory.read(p, q_pool, topk=None)  # [B,d], [B,1], [B,M]
#                 # 门控
#                 gate = torch.sigmoid(self.gate_proj(torch.cat([q_pool, sim_max], dim=-1)) + self.gate_bias)  # [B,d]
#                 fuse = self.mem_fuse(torch.cat([q_pool, m_read], dim=-1)) * self.mem_scale                  # [B,d]

#                 # 仅在极端 patch 上强融合（mask 控制）
#                 mask = extreme_mask[:, p].float().unsqueeze(-1)  # [B,1]
#                 gamma = 1.0 + gate * mask
#                 beta  = gate * mask * fuse
#                 q_new = gamma * q_pool + beta                    # [B,d]
#                 fused_list.append(q_new.unsqueeze(1))            # [B,1,d]

#                 # 训练阶段写入
#                 if self.training:
#                     write_mask = extreme_mask[:, p]  # bool
#                     if write_mask.any():
#                         self.memory.write(
#                             p,
#                             k_batch=q_pool[write_mask].detach(),
#                             v_batch=q_new[write_mask].detach(),
#                             w_batch=None
#                         )
#             patch_reps = torch.cat(fused_list, dim=1)  # [B, P, d]
#         # else: 不用记忆库，直接用 patch_reps

#         # ---------- Inter-patch attention（全局，真正做在 [B,P,d] 上） ----------
#         inter = patch_reps
#         for block in self.inter_patch_blocks:
#             inter = block(inter, attn_mask=inter_patch_mask)  # [B, P, d]

#         # ---------- 反 patchify：把 patch 级上下文广播回 token，并与局部表示融合 ----------
#         patch_ctx = inter.unsqueeze(2).expand(B, self.num_patches, self.patch_len, inter.size(-1))  # [B,P,pl,d]
#         patch_ctx = patch_ctx.reshape(B, -1, inter.size(-1))                                        # [B,P*pl,d]
#         patches_fused = patches_local + patch_ctx                                                      # [B,P*pl,d]

#         # ---------- 输出 ----------
#         y = self.proj_out(patches_fused)     # [B, total, 1]
#         y = y[:, -self.pred_len:, :]        # [B, pred_len, 1]

#         # ---------- RevIN inverse ----------
#         if self.revin:
#             mean, std = stats  # [B,1,1]
#             y = y * std.expand(-1, y.size(1), -1) + mean.expand(-1, y.size(1), -1)

#         if self.training:
#             self._global_step += 1

#         return y  # [B, pred_len, 1]

#Author  :   mkw 
#Time    :   2025/09/12 19:31:08
#Desc    :   None

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.embedding import DataEmbedding


def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    """
    生成因果窗口掩码。用于限制每个位置仅能与前面的窗口大小范围内的元素进行注意力计算。
    """
    bad = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
    for i in range(seq_len):
        left = max(0, i - win_size + 1)
        bad[i, :left] = True
    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_bias.masked_fill_(bad, torch.finfo(attn_bias.dtype).min)
    return attn_bias


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or (d_model * 4)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):  # x: [B, L, d]
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)[0]
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


# =========================
# Bucketed Extreme Memory (per patch position)
# =========================
class BucketedExtremeMemory(nn.Module):
    """
    每个 patch 位置一个桶：
    - keys[p]: [M, d]
    - vals[p]: [M, d]
    """
    def __init__(self, num_buckets: int, d_model: int, mem_size_per_bucket: int = 256,
                 topk: int = 16, temperature: float = 0.5, ema_momentum: float = 0.2, device=None):
        super().__init__()
        self.P = num_buckets
        self.d = d_model
        self.M = mem_size_per_bucket
        self.K = topk
        self.tau = temperature
        self.momentum = ema_momentum
        self.device = device

        self.keys = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])
        self.vals = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])
        self.register_buffer('ptr', torch.zeros(self.P, dtype=torch.long))
        self.ema_only = True

    @torch.no_grad()
    def write(self, p: int, k_batch: torch.Tensor, v_batch: torch.Tensor, w_batch: torch.Tensor = None):
        if k_batch.numel() == 0:
            return
        k_batch = F.normalize(k_batch, dim=-1)
        v_batch = F.normalize(v_batch, dim=-1)

        if self.ema_only:
            if w_batch is None:
                direction = k_batch.mean(dim=0, keepdim=True)   # [1,d]
                value_dir = v_batch.mean(dim=0, keepdim=True)   # [1,d]
            else:
                ws = w_batch
                if ws.dim() == 1: ws = ws.unsqueeze(-1)
                ws = ws / (ws.sum(dim=0, keepdim=True) + 1e-8)
                direction = (ws.t() @ k_batch).mean(dim=0, keepdim=True)
                value_dir = (ws.t() @ v_batch).mean(dim=0, keepdim=True)
            new_k = F.normalize((1 - self.momentum) * self.keys[p].data + self.momentum * direction, dim=-1)
            new_v = F.normalize((1 - self.momentum) * self.vals[p].data + self.momentum * value_dir, dim=-1)
            self.keys[p].data = new_k
            self.vals[p].data = new_v
        else:
            b = k_batch.size(0)
            start = int(self.ptr[p].item())
            idx = (torch.arange(b, device=k_batch.device) + start) % self.M
            self.keys[p].data[idx] = k_batch
            self.vals[p].data[idx] = v_batch
            self.ptr[p] = (self.ptr[p] + b) % self.M

    def read(self, p: int, q: torch.Tensor, topk=None):
        if topk is None:
            topk = self.K
        Kmat = F.normalize(self.keys[p], dim=-1)       # [M,d]
        qn = F.normalize(q, dim=-1)                    # [B,d]
        sim = (qn @ Kmat.t()) / self.tau               # [B,M]
        k = min(topk, self.M)
        topv, topi = torch.topk(sim, k=k, dim=-1)      # [B,k],[B,k]
        w = torch.softmax(topv, dim=-1)                # [B,k]
        Vmat = F.normalize(self.vals[p], dim=-1)       # [M,d]
        picked = Vmat[topi]                             # [B,k,d]
        m = (w.unsqueeze(-1) * picked).sum(dim=1)      # [B,d]
        s, _ = sim.max(dim=-1, keepdim=True)           # [B,1]

        wK = torch.zeros_like(sim)
        wK.scatter_(-1, topi, w)
        return m, s, wK


class PatchExtremeMemoryTransformer(nn.Module):
    """
    输入:  x ∈ [B, seq_len, 1]
    输出:  y ∈ [B, pred_len, 1]
    结构:
      - 窗口级样本 -> 切 P 个 patch
      - patch内 Transformer 堆叠
      - patch间 Transformer 堆叠（token = 每个 patch 的平均 token）
      - 分桶记忆 + 门控融合（仅在极端 patch 强融合）
      - （改造）并行浅全局分支 + Cross-Attention 回写
      - 反patchify回 token, 线性投影 -> 输出
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        d_model: int,
        revin: bool,
        num_heads: int,
        num_layers_in_patch: int = 2,
        num_layers_inter_patch: int = 1,
        d_ff: int = None,
        dropout: float = 0.1,
        mem_size_per_bucket: int = 256,
        mem_topk: int = 16,
        mem_tau: float = 0.5,
        mem_momentum: float = 0.2,
        z_thresh: float = 2.0,
        diff_z_thresh: float = 2.0,
        use_diff: bool = True,
        warmup_mem_steps: int = 200,
        warmup_ratio: float = 0.10,
        force_top1_steps: int = 50,
    ):
        super().__init__()
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.patch_len = patch_len
        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len

        # Embedding + 预投影到 total_len
        self.embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=dropout)
        self.predict_linear = nn.Linear(seq_len, self.total_len)  # [B,d,seq] -> [B,d,total]

        # patch 内 Transformer 堆叠
        self.in_patch_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers_in_patch)
            ]) for _ in range(self.num_patches)
        ])

        # patch 间 Transformer 堆叠（主干）
        self.inter_patch_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_inter_patch)
        ])

        self.post_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, 1)

        # 记忆（按位置分桶）
        self.use_memory = False
        self.memory = BucketedExtremeMemory(
            num_buckets=self.num_patches, d_model=d_model,
            mem_size_per_bucket=mem_size_per_bucket, topk=mem_topk,
            temperature=mem_tau, ema_momentum=mem_momentum
        )
        self.mem_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid()
        )
        self.mem_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.zeros(1))

        # 极端判定 & 训练暖身
        self.z_thresh = z_thresh
        self.diff_z_thresh = diff_z_thresh
        self.use_diff = use_diff
        self.ext_percentile = 0.9
        self.warmup_mem_steps = warmup_mem_steps
        self.warmup_ratio = warmup_ratio
        self.force_top1_steps = force_top1_steps
        self._global_step = 0

        # =========================
        # （新增）并行浅全局分支 + Cross-Attention 回写
        # =========================

        # 并行浅分支：从原始 tokens 直接聚合为 [B,P,d]
        # 采用可学习聚合：对每个 patch 的 token 维 pl 做线性池化
        self.token_pool_lin = nn.Linear(self.patch_len, 1, bias=False)  # 作用在 rearrange 后的最后一维(pl)

        # 浅分支的 inter-patch 块（结构与主干一致，深拷贝一套）
        self.inter_patch_blocks_shallow = nn.ModuleList(
            [copy.deepcopy(b) for b in self.inter_patch_blocks]
        )

        # 融合主干与浅分支的全局表示（concat -> MLP）
        self.global_fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Cross-Attention：让 token 级表示从 patch 级全局表示中按权回写
        self.token_global_xattn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.token_global_proj = nn.Linear(d_model, d_model)  # 可选
        self.token_dropout = nn.Dropout(dropout)

        # 可选门控：仅在极端 token 上加强全局注入
        self.enable_extreme_gate = True
        self.cross_gate_proj = nn.Linear(2 * d_model, 1)

    # ---------- masks ----------
    def _build_masks(self, device, dtype=torch.float32):
        intra_patch_mask = generate_causal_window_mask(self.patch_len, self.patch_len, device, dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, device, dtype)
        inter_mask_full = torch.zeros(self.total_len, self.total_len, dtype=dtype, device=device)
        inter_mask_full[:self.num_patches, :self.num_patches] = inter_mask
        return intra_patch_mask, inter_mask_full

    # ---------- 极端 patch 掩码 ----------
    def _patch_extreme_mask(self, x_raw, means, stdev):
        B, L_in, C = x_raw.shape
        T = self.total_len
        device = x_raw.device

        z = (x_raw - means) / (stdev + 1e-5)
        z_abs = z.abs().amax(dim=-1)  # [B, L_in]

        if L_in < T:
            pad = torch.zeros(B, T - L_in, device=device, dtype=z_abs.dtype)
            z_abs_full = torch.cat([z_abs, pad], dim=1)
        else:
            z_abs_full = z_abs[:, :T]

        z_abs_patch = rearrange(z_abs_full, 'b (np pl) -> b np pl',
                                np=self.num_patches, pl=self.patch_len)
        z_per_patch = z_abs_patch.amax(dim=-1)  # [B, P]
        z_flag = (z_per_patch > self.z_thresh)

        d_flag = torch.zeros_like(z_flag)
        if self.use_diff:
            if L_in >= 2:
                diff = x_raw[:, 1:, :] - x_raw[:, :-1, :]
                diff_z = diff / (stdev + 1e-5)
                diff_abs = diff_z.abs().amax(dim=-1)  # [B, L_in-1]
                diff_abs = torch.cat([torch.zeros(B, 1, device=device, dtype=diff_abs.dtype), diff_abs], dim=1)
            else:
                diff_abs = torch.zeros(B, L_in, device=device, dtype=z_abs.dtype)

            if L_in < T:
                pad2 = torch.zeros(B, T - L_in, device=device, dtype=diff_abs.dtype)
                diff_full = torch.cat([diff_abs, pad2], dim=1)
            else:
                diff_full = diff_abs[:, :T]

            diff_patch = rearrange(diff_full, 'b (np pl) -> b np pl',
                                   np=self.num_patches, pl=self.patch_len)
            d_per_patch = diff_patch.amax(dim=-1)
            d_flag = (d_per_patch > self.diff_z_thresh)

        flag = (z_flag | d_flag)

        if (~flag).all():
            strength = z_per_patch
            if self.use_diff:
                strength = torch.maximum(strength, d_per_patch)
            q = torch.quantile(strength, self.ext_percentile, dim=1, keepdim=True)
            flag = (strength >= q)

        if (~flag).all() and self._global_step < self.warmup_mem_steps:
            strength = z_per_patch
            if self.use_diff:
                strength = torch.maximum(strength, d_per_patch)
            k = max(1, int(self.num_patches * self.warmup_ratio))
            topk = torch.topk(strength, k=k, dim=1).indices
            flag = torch.zeros_like(strength, dtype=torch.bool)
            flag.scatter_(1, topk, True)

        if (~flag).all() and self._global_step < self.force_top1_steps:
            strength = z_per_patch
            if self.use_diff:
                strength = torch.maximum(strength, d_per_patch)
            top1 = torch.topk(strength, k=1, dim=1).indices
            flag = torch.zeros_like(strength, dtype=torch.bool)
            flag.scatter_(1, top1, True)

        return flag  # [B, P] bool
    

    def forward(self, x, x_mark=None,):  # x: [B, seq_len, 1]
        B, L, C = x.shape
        assert L == self.seq_len and C == 1

        x_raw = x  # 用于极端掩码评估

        # ---------- RevIN ----------
        if self.revin:
            mean = x.mean(1, keepdim=True).detach()  # [B,1,1]
            x_centered = x - mean
            std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,1]
            x = x_centered / std
            stats = (mean, std)
        else:
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            stats = (mean, std)

        # ---------- Embedding ----------
        x_emb = self.embedding(x)                          # [B, L, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')         # [B, d, L]
        x_emb = self.predict_linear(x_emb)                 # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')         # [B, total_len, d]

        # ---------- Patchify ----------
        patches = rearrange(x_emb, 'b (np pl) d -> b np pl d',
                            np=self.num_patches, pl=self.patch_len)  # [B,P,pl,d]

        # ---------- Masks ----------
        intra_patch_mask = generate_causal_window_mask(
            seq_len=self.patch_len, win_size=self.patch_len,
            device=patches.device, dtype=patches.dtype
        )  # [pl,pl]
        inter_patch_mask = generate_causal_window_mask(
            seq_len=self.num_patches, win_size=self.num_patches,
            device=patches.device, dtype=patches.dtype
        )  # [P,P]

        # ---------- Intra-patch attention（局部） ----------
        patch_outs = []
        for p in range(self.num_patches):
            out = patches[:, p]  # [B, pl, d]
            for block in self.in_patch_blocks[p]:
                out = block(out, attn_mask=intra_patch_mask)     # [B, pl, d]
            patch_outs.append(out)
        patches_local = torch.cat(patch_outs, dim=1)        # [B, P*pl, d]

        # ---------- Patch-level 聚合（主干） ----------
        patch_reps = patches_local.view(B, self.num_patches, self.patch_len, -1).mean(dim=2)  # [B,P,d]

        # ---------- Extreme mask ----------
        mean_for_mask, std_for_mask = stats
        extreme_mask = self._patch_extreme_mask(x_raw, mean_for_mask, std_for_mask)  # [B,P] bool

        # ---------- 分桶记忆 + 门控融合（在 patch token 上进行） ----------
        if self.use_memory:
            fused_list = []
            for p in range(self.num_patches):
                q_pool = patch_reps[:, p, :]                               # [B, d]
                m_read, sim_max, _ = self.memory.read(p, q_pool, topk=None)  # [B,d],[B,1]
                gate = torch.sigmoid(self.gate_proj(torch.cat([q_pool, sim_max], dim=-1)) + self.gate_bias)  # [B,d]
                fuse = self.mem_fuse(torch.cat([q_pool, m_read], dim=-1)) * self.mem_scale                  # [B,d]

                mask = extreme_mask[:, p].float().unsqueeze(-1)  # [B,1]
                gamma = 1.0 + gate * mask
                beta  = gate * mask * fuse
                q_new = gamma * q_pool + beta                    # [B,d]
                fused_list.append(q_new.unsqueeze(1))            # [B,1,d]

                if self.training:
                    write_mask = extreme_mask[:, p]
                    if write_mask.any():
                        self.memory.write(
                            p,
                            k_batch=q_pool[write_mask].detach(),
                            v_batch=q_new[write_mask].detach(),
                            w_batch=None
                        )
            patch_reps = torch.cat(fused_list, dim=1)  # [B, P, d]

        # ============================================================
        # （新增）并行浅全局分支 + 与主干的全局表征融合
        # ============================================================
        # 浅分支：从原始 tokens 直接聚合得到 [B,P,d]
        # rearrange: [B,P,pl,d] -> [B,P,d,pl] -> Linear(pl->1) -> [B,P,d,1] -> squeeze -> [B,P,d]
        shallow_reps = self.token_pool_lin(rearrange(patches, 'b p pl d -> b p d pl')).squeeze(-1)

        # 跑一套浅分支的 inter-patch
        inter_shallow = shallow_reps
        for block in self.inter_patch_blocks_shallow:
            inter_shallow = block(inter_shallow, attn_mask=inter_patch_mask)  # [B,P,d]

        # 主干分支的 inter-patch
        inter_main = patch_reps
        for block in self.inter_patch_blocks:
            inter_main = block(inter_main, attn_mask=inter_patch_mask)        # [B,P,d]

        # 融合两个全局分支
        inter = self.global_fuse(torch.cat([inter_main, inter_shallow], dim=-1))  # [B,P,d]

        # ============================================================
        # （改造）Cross-Attention 回写到 token 级（替代均匀广播）
        # ============================================================
        tokens = patches_local         # [B,P*pl,d]
        global_ctx = inter             # [B,P,d]

        tokens_refined, _ = self.token_global_xattn(
            query=tokens, key=global_ctx, value=global_ctx, need_weights=False
        )  # [B,P*pl,d]
        tokens_refined = self.token_global_proj(tokens_refined)
        tokens_refined = self.token_dropout(tokens_refined)

        if self.enable_extreme_gate:
            # 将 [B,P] 的极端掩码扩展到 token 维 [B,P*pl,1]
            ext_mask_tok = extreme_mask.unsqueeze(-1).expand(-1, -1, self.patch_len)   # [B,P,pl]
            ext_mask_tok = ext_mask_tok.reshape(B, -1).unsqueeze(-1)                   # [B,P*pl,1]
            # data-dependent gate \in (0,1)
            gate = torch.sigmoid(self.cross_gate_proj(torch.cat([tokens, tokens_refined], dim=-1)))  # [B,P*pl,1]
            gate = gate * ext_mask_tok
            patches_fused = tokens + gate * tokens_refined
        else:
            patches_fused = tokens + tokens_refined                                     # [B,P*pl,d]

        # ---------- 输出 ----------
        y = self.proj_out(patches_fused)     # [B, total, 1]
        y = y[:, -self.pred_len:, :]         # [B, pred_len, 1]

        # ---------- RevIN inverse ----------
        if self.revin:
            mean, std = stats  # [B,1,1]
            y = y * std.expand(-1, y.size(1), -1) + mean.expand(-1, y.size(1), -1)

        if self.training:
            self._global_step += 1

        return y  # [B, pred_len, 1]

