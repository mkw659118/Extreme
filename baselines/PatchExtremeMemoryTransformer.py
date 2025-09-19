#Author  :   mkw 
#Time    :   2025/09/17 10:50:52
#Desc    :   None

import torch
import torch.nn as nn
from einops import rearrange
from layers.embedding import DataEmbedding
from layers.memory import BucketedExtremeMemory



def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    win_size = seq_len // 2
    # 上三角掩码
    upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
    # 左侧窗口注意
    for i in range(seq_len):
        left = max(0, i - win_size + 1)
        upper[i, :left] = True
    # 全零矩阵
    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    # 左上角填充-inf
    attn_bias.masked_fill_(upper, torch.finfo(attn_bias.dtype).min)

   
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


class PatchExtremeMemoryTransformer(nn.Module):
    """
    PathFormer-style：Patch Division -> {Intra-branch || Inter-branch} 并行 -> Fusion

    输入:  x ∈ [B, seq_len, 1]
    输出:  y ∈ [B, pred_len, 1]

    关键改动（相对你原始版本）：
      - 不再“先 intra 再 inter”，而是把 patch 切好后**并行**：
          * Intra-branch：每个 patch 内做 Transformer 堆叠
          * Inter-branch：对每个 patch 可学习池化 -> [B,P,d]，在 patch 序列上做 Transformer
      - Fusion：用 Cross-Attention 把 inter 的全局表征回写到 token 级，与 intra 的输出融合
      - 极端分桶记忆：仍然作用于 patch 级（inter 分支的 patch 表征），仅在“极端 patch”时强融合/写入
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        d_model: int,
        win_size: int,
        revin: bool,
        num_heads: int,
        num_layers_intra_patch: int = 1,
        num_layers_inter_patch: int = 1,
        d_ff: int = None,
        dropout: float = 0.1,
        # ---- 记忆参数 ----
        mem_size_per_bucket: int = 256,
        mem_topk: int = 16,
        mem_tau: float = 0.5,
        mem_momentum: float = 0.2,
        # ---- 极端检测 ----
        z_thresh: float = 2.0,
        diff_z_thresh: float = 2.0,
        use_diff: bool = True,
        warmup_mem_steps: int = 200,
        warmup_ratio: float = 0.10,
        force_top1_steps: int = 50,
        # ---- 其他 ----
        enable_memory: bool = True,
        enable_extreme_gate: bool = True,
    ):
        super().__init__()

        # --------- 基本长度与 patch 划分 ----------
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.patch_len = patch_len
        self.win_size = win_size
        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len  # 记作 P

        # --------- Embedding & 预投影到 total_len ----------
        self.embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=dropout)
        # 把 seq 维投影到 total_len（预测期也以 token 形式进入 encoder）
        self.predict_linear = nn.Linear(seq_len, self.total_len)  # [B,d,seq] -> [B,d,total]

        # =====================================================
        # 分支 A：Intra-branch（patch 内）—— 每个 patch 上的 token 级 Transformer
        #  结构：P 份、每份堆 num_layers_in_patch 个 TransformerBlock（权重不共享）
        # =====================================================
        self.in_patch_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers_intra_patch)
            ])
            for _ in range(self.num_patches)
        ])

        # =====================================================
        # 分支 B：Inter-branch（patch 间）—— patch-token 序列上的 Transformer
        #  步骤：
        #   1) 可学习池化：把每个 patch 的 pl 个 token -> 1 个 patch-token
        #   2) 在 [B,P,d] 上堆 num_layers_inter_patch 个 TransformerBlock（权重共享）
        # =====================================================
        # 1) 可学习池化：对每个 patch 的长度维 pl 做线性池化（权重学习）
        self.token_pool_lin = nn.Linear(self.patch_len, 1, bias=False)  # 作用在 rearrange 后的最后一维(pl)

        # 2) patch 序列上的 Transformer 堆叠
        self.inter_patch_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_inter_patch)
        ])

        # --------- 融合层（把 inter 的全局上下文回写到 token 级） ----------
        self.fusion_xattn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.fusion_proj = nn.Linear(d_model, d_model)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enable_extreme_gate = enable_extreme_gate
        self.cross_gate_proj = nn.Linear(2 * d_model, 1)  # 用于极端 token 的门控强注入

        # --------- 输出头 ----------
        self.post_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, 1)

        # =====================================================
        # （可选）极端分桶记忆：只在“patch 级”进行（即作用于 inter-branch 的 patch-token）
        # =====================================================
        self.use_memory = enable_memory
        if self.use_memory:
            self.memory = BucketedExtremeMemory(
                num_buckets=self.num_patches, d_model=d_model,
                mem_size_per_bucket=mem_size_per_bucket, topk=mem_topk,
                temperature=mem_tau, ema_momentum=mem_momentum
            )
            # 记忆读出后的融合（v 与 q 的融合）
            self.mem_fuse = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            # 根据相似度强弱出门控系数（逐通道）
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.Sigmoid()
            )
            self.mem_scale = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.zeros(1))

        # --------- 极端 patch 判定 & 训练暖身 ----------
        self.z_thresh = z_thresh
        self.diff_z_thresh = diff_z_thresh
        self.use_diff = use_diff
        self.ext_percentile = 0.9
        self.warmup_mem_steps = warmup_mem_steps
        self.warmup_ratio = warmup_ratio
        self.force_top1_steps = force_top1_steps
        self._global_step = 0

        # ============= 极端 patch 掩码 =============
    def patch_extreme_mask(self, x_raw, mean, std):
        """
        返回: flag ∈ [B,P] 的 bool 掩码，标出“极端”的 patch。
        逻辑：基于值域 z-score 与一阶差分 z-score 的最大值；结合暖身/分位数兜底。
        """
        B, L, _ = x_raw.shape                 # B=批大小，L=输入长度，D=通道数（通常=1）
        T = self.total_len                    # 统一对齐的序列总长度（可能包含padding）
        device = x_raw.device                 # 当前数据所在的设备（CPU/GPU）

        # ---------- Step 1: 值域 z-score ----------
        z = (x_raw - mean) / (std + 1e-5)     # 标准化，防止除0：得到逐点 z-score，[B,L,D]
        z_abs = z.abs().amax(dim=-1)          # 取绝对值并在通道维求最大值 → [B,L]

        # ---------- Step 2: 补齐到 total_len ----------
        if L < T:                          # 如果输入长度比总长度短 → 补零
            pad = torch.zeros(B, T - L, device=device, dtype=z_abs.dtype)
            z_abs_full = torch.cat([z_abs, pad], dim=1)   # 拼接得到完整长度 [B,T]
        else:                                 # 如果比 T 长 → 截断
            z_abs_full = z_abs[:, :T]

        # ---------- Step 3: 聚合到 patch ----------
        # [B,num_patches,patch_len]  把序列划分成 patch 结构
        z_abs_patch = rearrange(z_abs_full, 'b (np pl) -> b np pl', np=self.num_patches, pl=self.patch_len)
        z_per_patch = z_abs_patch.amax(dim=-1)  # 每个 patch 内取最大值 [B,P]
        z_flag = (z_per_patch > self.z_thresh)  # 超过阈值的 patch 标记为 True

        # ---------- Step 4: 一阶差分检测突变 ----------
        d_flag = torch.zeros_like(z_flag)     # 初始化差分掩码 [B,P]
        if self.use_diff:                     # 如果启用差分检测
            if L >= 2:                     # 序列长度足够长
                diff = x_raw[:, 1:, :] - x_raw[:, :-1, :]  # 一阶差分 [B,L-1,C]
                diff_z = diff / (std + 1e-5)               # 标准化
                diff_abs = diff_z.abs().amax(dim=-1)       # 按通道取最大 [B,L-1]
                diff_abs = torch.cat([                   # 开头补0对齐
                    torch.zeros(B, 1, device=device, dtype=diff_abs.dtype),
                    diff_abs
                ], dim=1)                                # [B,L]
            else:
                diff_abs = torch.zeros(B, L, device=device, dtype=z_abs.dtype)

            if L < T:                                # 补齐到 T
                pad2 = torch.zeros(B, T - L, device=device, dtype=diff_abs.dtype)
                diff_full = torch.cat([diff_abs, pad2], dim=1)  # [B,T]
            else:
                diff_full = diff_abs[:, :T]

            diff_patch = rearrange(                     # 划分成 patch
                diff_full, 'b (np pl) -> b np pl',
                np=self.num_patches, pl=self.patch_len
            )                                           # [B,P,pl]
            d_per_patch = diff_patch.amax(dim=-1)       # 每个 patch 内最大差分值 [B,P]
            d_flag = (d_per_patch > self.diff_z_thresh) # 超过阈值则置 True

        # ---------- Step 5: 综合值域与差分 ----------
        flag = (z_flag | d_flag)              # 任意一个满足条件即为极端 patch [B,P]

        # ---------- Step 6: 兜底策略 ----------
        # (1) 如果没有任何 patch 被标记
        if (~flag).all():
            strength = z_per_patch            # 强度 = 值域极值
            if self.use_diff:                 # 如果用差分，也纳入计算
                strength = torch.maximum(strength, d_per_patch)
            q = torch.quantile(               # 按分位数取阈值
                strength, self.ext_percentile, dim=1, keepdim=True
            )
            flag = (strength >= q)            # 大于分位数阈值的 patch 标记为 True

        # (2) 如果仍然没有，且训练步数 < warmup 阶段
        if (~flag).all() and self._global_step < self.warmup_mem_steps:
            strength = z_per_patch
            if self.use_diff:
                strength = torch.maximum(strength, d_per_patch)
            k = max(1, int(self.num_patches * self.warmup_ratio))  # Top-k
            topk = torch.topk(strength, k=k, dim=1).indices        # 取最强的 k 个 patch
            flag = torch.zeros_like(strength, dtype=torch.bool)    # 重新建掩码
            flag.scatter_(1, topk, True)                           # Top-k 标 True

        # (3) 如果仍然没有，且步数 < 强制 top1 阶段
        if (~flag).all() and self._global_step < self.force_top1_steps:
            strength = z_per_patch
            if self.use_diff:
                strength = torch.maximum(strength, d_per_patch)
            top1 = torch.topk(strength, k=1, dim=1).indices        # 取最强的 1 个 patch
            flag = torch.zeros_like(strength, dtype=torch.bool)
            flag.scatter_(1, top1, True)                           # 该 patch 标 True

        return flag  # 输出 [B,P] 的 bool 掩码，标记哪些 patch 是“极端的”


    def forward(self, x, x_mark=None):
        """
        x: [B, seq_len, 1]
        return y: [B, pred_len, 1]
        """
        B, L, C = x.shape
        assert L == self.seq_len and C == 1

        x_raw = x  # 极端掩码基于原始值计算

        # ---------- RevIN ----------
        if self.revin:
            mean = x.mean(1, keepdim=True).detach()  # [B,1,1]
            x_centered = x - mean
            std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x_centered / std
            stats = (mean, std)
        else:
            mean = x.mean(1, keepdim=True).detach()
            std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            stats = (mean, std)

        # ---------- Token embedding 并预投影到 total_len ----------
        x_emb = self.embedding(x)                      # [B, L, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')     # [B, d, L]
        x_emb = self.predict_linear(x_emb)             # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')     # [B, total_len, d]

        # ---------- Patch Division ----------
        # patches: [B, P, pl, d]  （P = total_len // pl）
        patches = rearrange(x_emb, 'b (p pl) d -> b p pl d',p=self.num_patches, pl=self.patch_len)

        # ---------- 构造 Mask ----------
        # Intra: 每个 patch 内因果 mask
        intra_patch_mask = generate_causal_window_mask(
            seq_len=self.patch_len, win_size=self.win_size,
            device=patches.device, dtype=patches.dtype
        )  # [pl, pl]

        # Inter: patch 序列上的因果 mask
        inter_patch_mask = generate_causal_window_mask(
            seq_len=self.num_patches, win_size=self.win_size,
            device=patches.device, dtype=patches.dtype
        )  # [P, P]

        # ---------- 极端 patch 掩码（用于门控 & memory 写条件） ----------
        mean, std = stats
        extreme_mask = self.patch_extreme_mask(x_raw, mean, std)  # [B,P] bool

        # =====================================================
        # 分支 A：Intra-branch
        # =====================================================
        intra_patch_outs = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]  # [B, pl, d]
            for block in self.in_patch_blocks[p]:
                out = block(out, attn_mask=intra_patch_mask)  # 仅在本 patch 内注意
            intra_patch_outs.append(out)
        intra_tokens = torch.cat(intra_patch_outs, dim=1)  # [B, P*pl, d] —— token 级

        # =====================================================
        # 分支 B：Inter-branch + Memory（可选）
        #   - inter 只在 P 维做注意力
        #   - memory 仅作为 inter 输入的“加性增量”
        # =====================================================
        inter_patches = patches  # [B,P,pl,d] 默认不变

        if self.use_memory:
            # 1) 轻量聚合得到每个 patch 的查询向量 q（仅用于检索）
            q_patch = patches.mean(dim=2)  # [B,P,d]

            q_new_list = []
            for p in range(self.num_patches):
                q = q_patch[:, p, :]  # [B,d]

                # 读记忆：按 patch 位置分桶
                m_read, sim_max, _ = self.memory.read(p, q, topk=None)  # m_read:[B,d], sim_max:[B,1]

                # 相似度感知门控（逐通道）
                gate = torch.sigmoid(self.gate_proj(torch.cat([q, sim_max], dim=-1)) + self.gate_bias)  # [B,d]
                fuse = self.mem_fuse(torch.cat([q, m_read], dim=-1)) * self.mem_scale                   # [B,d]

                # 仅在极端 patch 强化
                mask = extreme_mask[:, p].float().unsqueeze(-1)  # [B,1]
                q_new = (1.0 + gate * mask) * q + (gate * mask) * fuse  # [B,d]
                q_new_list.append(q_new.unsqueeze(1))  # [B,1,d]

                # 训练期只对极端 patch 写回
                if self.training:
                    write_mask = extreme_mask[:, p]
                    if write_mask.any():
                        self.memory.write(
                            p,
                            k_batch=q[write_mask].detach(),
                            v_batch=q_new[write_mask].detach(),
                            w_batch=None
                        )
            q_new_all = torch.cat(q_new_list, dim=1)          # [B,P,d]
            delta = (q_new_all - q_patch).unsqueeze(2)         # [B,P,1,d]
            inter_patches = patches + delta.expand(-1, -1, self.patch_len, -1)  # [B,P,pl,d]

        # 对每个“位置 t”，沿 P维度 做 inter 注意力 —— #
        inter_patches = rearrange(inter_patches, 'B P pl d -> B pl P d')
        inter_patches = rearrange(inter_patches, 'B pl P d -> (B pl) P d')

        for block in self.inter_patch_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_patch_mask)          # [B*pl, P, d]

        # === 还原到 token 级：[B*pl,P,d] -> [B,P*pl,d]，用于与 intra_tokens 融合 ===
        inter_tokens = rearrange(inter_patches, '(B pl) P d -> B (P pl) d', B=B, pl=self.patch_len)  # [B,P*pl,d]

        # =====================================================
        # 融合：intra 与 inter 相加（可选极端门控）
        # =====================================================
        if self.enable_extreme_gate:
            # 把 [B,P] 的极端掩码扩展到 token 级 [B,P*pl,1]
            ext_mask_tok = extreme_mask.unsqueeze(-1).expand(-1, -1, self.patch_len)  # [B,P,pl]
            ext_mask_tok = ext_mask_tok.reshape(B, -1).unsqueeze(-1)                  # [B,P*pl,1]
            # data-dependent gate ∈ (0,1)
            gate = torch.sigmoid(self.cross_gate_proj(torch.cat([intra_tokens, inter_tokens], dim=-1)))  # [B,P*pl,1]
            gate = gate * ext_mask_tok
            final = intra_tokens + gate * inter_tokens
        else:
            final = intra_tokens + inter_tokens

        final = self.post_norm(final)  # [B, total_len, d], 对特征做归一化处理

        # ---------- 输出头 ----------
        y = self.proj_out(final)         # [B, total_len, 1], 投影回原始特征维度
        y = y[:, -self.pred_len:, :]     # [B, pred_len, 1]

        # ---------- RevIN inverse ----------
        if self.revin:
            mean, std = stats
            y = y * std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + mean[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        if self.training:
            self._global_step += 1

        return y
