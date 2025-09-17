#Author  :   mkw 
#Time    :   2025/09/17 10:50:52
#Desc    :   None

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        num_layers_in_patch: int = 2,
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
        enable_memory: bool = False,
        enable_extreme_gate: bool = False,
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
                for _ in range(num_layers_in_patch)
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

    # ============= 极端 patch 掩码：复用你现有逻辑（轻调） =============
    def _patch_extreme_mask(self, x_raw, means, stdev):
        """
        返回: flag ∈ [B,P] 的 bool 掩码，标出“极端”的 patch。
        逻辑：基于值域 z-score 与一阶差分 z-score 的最大值；结合暖身/分位数兜底。
        """
        B, L_in, C = x_raw.shape
        T = self.total_len
        device = x_raw.device

        # 标准化
        z = (x_raw - means) / (stdev + 1e-5)             # [B,L,1]
        z_abs = z.abs().amax(dim=-1)                     # [B,L]

        # 对齐到 total_len（后段为 0 填充）
        if L_in < T:
            pad = torch.zeros(B, T - L_in, device=device, dtype=z_abs.dtype)
            z_abs_full = torch.cat([z_abs, pad], dim=1)
        else:
            z_abs_full = z_abs[:, :T]

        # 汇聚到 patch
        z_abs_patch = rearrange(z_abs_full, 'b (np pl) -> b np pl',
                                np=self.num_patches, pl=self.patch_len)
        z_per_patch = z_abs_patch.amax(dim=-1)           # [B,P]
        z_flag = (z_per_patch > self.z_thresh)

        # 一阶差分作为“突变”信号
        d_flag = torch.zeros_like(z_flag)
        if self.use_diff:
            if L_in >= 2:
                diff = x_raw[:, 1:, :] - x_raw[:, :-1, :]
                diff_z = diff / (stdev + 1e-5)
                diff_abs = diff_z.abs().amax(dim=-1)     # [B,L-1]
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
            d_per_patch = diff_patch.amax(dim=-1)        # [B,P]
            d_flag = (d_per_patch > self.diff_z_thresh)

        flag = (z_flag | d_flag)

        # 兜底：按强度 top-q / top-k / top-1
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

        return flag  # [B,P] bool

    # ============================== 前向 ==============================
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
        x_emb = self.embedding(x)                 # [B, L, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')  # [B, d, L]
        x_emb = self.predict_linear(x_emb)        # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')  # [B, total_len, d]

        # ---------- Patch Division ----------
        # patches: [B, P, pl, d]  （P = total_len // pl）
        patches = rearrange(x_emb, 'b (p pl) d -> b p pl d', p=self.num_patches, pl=self.patch_len)

        # ---------- 构造 Mask ----------
        # Intra-branch：每个 patch 内因果 mask（窗口=pl，等价全因果）
        intra_patch_mask = generate_causal_window_mask(
            seq_len=self.patch_len, win_size=self.win_size,
            device=patches.device, dtype=patches.dtype
        )  # [pl,pl]

        # Inter-branch：patch 序列上的因果 mask（窗口=P，等价全因果）
        inter_patch_mask = generate_causal_window_mask(
            seq_len=self.num_patches, win_size=self.win_size,
            device=patches.device, dtype=patches.dtype
        )  # [P,P]

        # ---------- 极端 patch 掩码（用于记忆与门控） ----------
        mean_for_mask, std_for_mask = stats
        extreme_mask = self._patch_extreme_mask(x_raw, mean_for_mask, std_for_mask)  # [B,P] bool

        # =====================================================
        # 分支 A：Intra-branch（patch 内并行）
        #  输入：[B,P,pl,d]，逐 patch 走各自的堆叠；输出保持 token 级：[B,P*pl,d]
        # =====================================================
        intra_patch_outs = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]                       # [B, pl, d]
            for block in self.in_patch_blocks[p]:
                out = block(out, attn_mask=intra_patch_mask)  # 仅在本 patch 内注意
            intra_patch_outs.append(out)
        intra_tokens = torch.cat(intra_patch_outs, dim=1)     # [B, P*pl, d] —— token 级输出

        # =====================================================
        # 分支 B：Inter-branch（patch 间并行）
        #  1) 可学习池化得到 patch-token：[B,P,d]
        #  2) 在 patch 序列上堆 Transformer
        #  3) （可选）对“极端 patch”融合记忆并写入
        # =====================================================
        # 1) 可学习池化： [B,P,pl,d] -> [B,P,d,pl] -> Linear(pl->1) -> [B,P,d,1] -> squeeze -> [B,P,d]
        patch_tokens = self.token_pool_lin(rearrange(patches, 'b p pl d -> b p d pl')).squeeze(-1)  # [B,P,d]

        # 2) 记忆读写（先读后写），只在 inter-branch 的 patch-token 上进行
        if self.use_memory:
            fused_list = []
            for p in range(self.num_patches):
                q = patch_tokens[:, p, :]  # [B,d]
                # 读
                m_read, sim_max, _ = self.memory.read(p, q, topk=None)  # m_read:[B,d], sim_max:[B,1]
                gate = torch.sigmoid(self.gate_proj(torch.cat([q, sim_max], dim=-1)) + self.gate_bias)  # [B,d]
                fuse = self.mem_fuse(torch.cat([q, m_read], dim=-1)) * self.mem_scale                  # [B,d]

                # 仅在极端 patch 上加强融合
                mask = extreme_mask[:, p].float().unsqueeze(-1)  # [B,1]
                q_new = (1.0 + gate * mask) * q + (gate * mask) * fuse
                fused_list.append(q_new.unsqueeze(1))            # [B,1,d]

                # 训练阶段：极端 patch 才写回
                if self.training:
                    write_mask = extreme_mask[:, p]
                    if write_mask.any():
                        self.memory.write(
                            p,
                            k_batch=q[write_mask].detach(),
                            v_batch=q_new[write_mask].detach(),
                            w_batch=None
                        )
            patch_tokens = torch.cat(fused_list, dim=1)  # [B,P,d]

        # 3) 在 patch 序列上做 Transformer（捕获跨 patch 的全局依赖）
        inter_ctx = patch_tokens
        for block in self.inter_patch_blocks:
            inter_ctx = block(inter_ctx, attn_mask=inter_patch_mask)  # [B,P,d]

        # =====================================================
        # Fusion：把 inter 的全局表征回写到 token 级，与 intra 的 token 输出融合
        #   - query: token 级（来自 intra-branch 的输出）
        #   - key/value: patch 级（来自 inter-branch 的输出）
        #   - 再加残差 & （可选）极端 token 门控强化
        # =====================================================
        tokens = intra_tokens                 # [B,P*pl,d]
        global_ctx = inter_ctx                # [B,P,d]

        # Cross-Attn 回写（token <- patch）
        tokens_refined, _ = self.fusion_xattn(
            query=tokens, key=global_ctx, value=global_ctx, need_weights=False
        )  # [B,P*pl,d]
        tokens_refined = self.fusion_proj(tokens_refined)
        tokens_refined = self.fusion_dropout(tokens_refined)

        if self.enable_extreme_gate:
            # 把 [B,P] 的极端掩码扩展到 token 级 [B,P*pl,1]
            ext_mask_tok = extreme_mask.unsqueeze(-1).expand(-1, -1, self.patch_len)  # [B,P,pl]
            ext_mask_tok = ext_mask_tok.reshape(B, -1).unsqueeze(-1)                  # [B,P*pl,1]
            # data-dependent gate ∈ (0,1)
            gate = torch.sigmoid(self.cross_gate_proj(torch.cat([tokens, tokens_refined], dim=-1)))  # [B,P*pl,1]
            gate = gate * ext_mask_tok
            fused_tokens = tokens + gate * tokens_refined
        else:
            fused_tokens = tokens + tokens_refined

        fused_tokens = self.post_norm(fused_tokens)  # 轻微稳态化

        # ---------- 输出头 ----------
        y = self.proj_out(fused_tokens)         # [B, total_len, 1]
        y = y[:, -self.pred_len:, :]            # [B, pred_len, 1]

        # ---------- RevIN inverse ----------
        if self.revin:
            mean, std = stats  # [B,1,1]
            y = y * std.expand(-1, y.size(1), -1) + mean.expand(-1, y.size(1), -1)

        if self.training:
            self._global_step += 1

        return y
