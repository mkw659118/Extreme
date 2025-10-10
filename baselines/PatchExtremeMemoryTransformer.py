#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer with Sample-level Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.embedding import DataEmbedding
from layers.memory import SampleMemory              # 样本级记忆库


class DFT(nn.Module):
    def __init__(self, top_k):
        super(DFT, self).__init__()
        self.top_k = top_k

    def forward(self, x, x_mark=None):
        # x: [B, L, D]
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0  # 去除直流分量
        topk_vals, _ = torch.topk(freq, self.top_k, dim=1)
        threshold = topk_vals[:, -1:, :]  # [B, 1, D]
        xf = torch.where(freq >= threshold, xf, torch.zeros_like(xf))
        seasonal = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        return seasonal  # [B, L, D]


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x, x_mark=None):
        # x: [B, L, D]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))  # [B, D, L]
        x = x.permute(0, 2, 1)  # [B, L, D]
        return x

def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    # 使传入的 win_size 生效；非法则回退为 seq_len // 2
    if win_size is None or win_size <= 0 or win_size > seq_len:
        win_size = max(1, seq_len // 2)

    # True=屏蔽, False=可见：上三角（因果）
    upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
    # 左侧窗口裁剪
    if win_size < seq_len:
        for i in range(seq_len):
            left = max(0, i - win_size + 1)
            upper[i, :left] = True

    # 加性掩码：可见=0，屏蔽=-inf
    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_bias.masked_fill_(upper, float('-inf'))
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
        x1 = self.norm1(x)
        y = self.attn(x1, x1, x1, attn_mask=attn_mask)[0]
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class PatchExtremeMemoryTransformer(nn.Module):
    """
    Patch Division -> {Intra-branch || Inter-branch} 并行 -> Gated Sum（无 cross-attn）
    记忆库：样本级（按 sample_id 聚合/检索），仅对“极端 patch”注入
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
        use_memory: bool,
        num_layers_intra_patch: int,
        num_layers_inter_patch: int,
        config,
        mem_size: int = 4096,       
        mem_topk: int = 16,
        mem_tau: float = 0.5,
        mem_momentum: float = 0.2
    ):
        super().__init__()
        self.config = config
        # --------- 基本长度与 patch 划分 ----------
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.d_model = d_model
        self.use_memory = use_memory
        self.num_layers_intra_patch = num_layers_intra_patch
        self.num_layers_inter_patch = num_layers_inter_patch
        self.patch_len = patch_len
        self.win_size = win_size
        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len  # P

        self.top_k = 8
        self.kernel_size = 25

        # --------- Embedding & 预投影到 total_len ----------
        self.embedding = DataEmbedding(c_in=8, d_model=d_model, dropout=0.1)
        self.predict_linear = nn.Linear(seq_len, self.total_len)  # [B,d,seq] -> [B,d,total]
        self.topk_to_total_linear = nn.Linear(mem_topk, self.total_len)
        # --------- 周期趋势分解 ----------
        self.dft = DFT(top_k=self.top_k)
        self.trend = moving_avg(kernel_size=self.kernel_size, stride=1)

        # =====================================================
        # 分支 A：Intra-branch（patch 内）
        # =====================================================
        self.intra_patch_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.1)
                for _ in range(self.num_layers_intra_patch)
            ])
            for _ in range(self.num_patches)
        ])

        # =====================================================
        # 分支 B：Inter-branch（patch 间）
        # =====================================================
        self.inter_patch_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.1)
            for _ in range(self.num_layers_inter_patch)
        ])

        # --------- 融合（门控加和） ----------
        self.cross_gate_proj = nn.Linear(2 * d_model, 1)

        # --------- 输出头 ----------
        self.post_norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, 1)

        # --------- 样本级记忆库 ----------
        if self.use_memory:
            self.memory = SampleMemory(
                d_model=d_model, mem_size=mem_size,
                topk=mem_topk, temperature=mem_tau, ema_momentum=mem_momentum
            )
            self.mem_fuse = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
            self.gate_proj = nn.Linear(2 * d_model, d_model)  # 从 1024 到 512
            self.mem_scale = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.zeros(1))

        # 掩码缓存
        self._cached_intra = None
        self._cached_inter = None

    # ============= 掩码缓存器 =============
    def get_intra_mask(self, device, dtype):
        # 构造唯一标识：掩码依赖于 device (CPU/GPU) 和 dtype (float32/float16)
        key = (device, dtype)

        # 从缓存里取出已有的 intra_patch 掩码
        m = self._cached_intra

        # 如果缓存为空，或者缓存的设备/精度不匹配，就重新生成
        if m is None or m['key'] != key:
            # 生成长度为 patch_len 的因果窗口掩码
            mask = generate_causal_window_mask(self.patch_len, self.win_size, device, dtype)
            # 把新的掩码和它的 key 缓存下来
            self._cached_intra = {'key': key, 'mask': mask}

        # 返回缓存里的掩码（如果已有且匹配就直接复用）
        return self._cached_intra['mask']


    def get_inter_mask(self, device, dtype):
        # 构造唯一标识：掩码依赖于 device 和 dtype
        key = (device, dtype)

        # 从缓存里取出已有的 inter_patch 掩码
        m = self._cached_inter

        # 如果缓存为空，或者缓存的设备/精度不匹配，就重新生成
        if m is None or m['key'] != key:
            # 生成长度为 num_patches 的因果窗口掩码
            mask = generate_causal_window_mask(self.num_patches, self.win_size, device, dtype)
            # 把新的掩码和它的 key 缓存下来
            self._cached_inter = {'key': key, 'mask': mask}

        # 返回缓存里的掩码
        return self._cached_inter['mask']


    def forward(self, x, x_mark=None, sample_ids=None):
        
        B, _, _ = x.shape
        print(x.shape)

        x_combined = torch.cat([x, x_mark], dim=-1)  # [B, L, C + C_mark]，C_mark是x_mark的特征数

        # ---------- RevIN ----------
        if self.revin:
            mean = x.mean(1, keepdim=True).detach()
            x_centered = x - mean
            std = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_normalized = x_centered / std
            # 拼接归一化后的核心特征与原始辅助特征（或对辅助特征单独处理）
            x_combined = torch.cat([x_normalized, x_mark], dim=-1)
            stats = (mean, std)
        
        # ---------- Embedding & 预投影 ----------
        x_emb = self.embedding(x_combined)             # [B, L, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')     # [B, d, L]
        x_emb = self.predict_linear(x_emb)             # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')     # [B, total_len, d]

        # seasonal = self.dft(x_emb)
        trend = self.trend(x_emb)
        residual = x_emb - trend # 将残差结果作为周期性特征

        # ---------- Patch Division ----------
        patches = rearrange(residual, 'b (p pl) d -> b p pl d', p=self.num_patches, pl=self.patch_len)

        # ---------- Mask（缓存） ----------
        intra_patch_mask = self.get_intra_mask(patches.device, patches.dtype)  # [pl, pl]
        inter_patch_mask = self.get_inter_mask(patches.device, patches.dtype)  # [P, P]

        # =====================================================
        # 分支 A：Intra-branch
        # =====================================================
        outs_intra = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]
            for block in self.intra_patch_blocks[p]:
                out = block(out, attn_mask=intra_patch_mask)
            outs_intra.append(out)
        intra_tokens = torch.cat(outs_intra, dim=1)  # [B, P*pl, d]

        # =====================================================
        # 分支 B：Inter-branch
        # =====================================================
        inter_patches = rearrange(patches, 'B P pl d -> (B pl) P d')

        for block in self.inter_patch_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_patch_mask)

        inter_tokens = rearrange(inter_patches, '(B pl) P d -> B (P pl) d', B=B, pl=self.patch_len)

        final = intra_tokens + inter_tokens
        
        final = self.post_norm(final)  # [B, total_len, d]
        
        
        # =====================================================
        # 使用 final 作为查询向量进行记忆库的存储和读取
        # =====================================================
    
        # ---------- 记忆库存入 ----------
        if self.use_memory:
            # 使用 final 作为查询向量进行记忆库操作
            q_s = final.sum(dim=1)  # [B, d]
            m_read, _ , _ = self.memory.read(sample_ids, q_s)  # 从记忆库读取

            # 扩展 q_s 使其与 m_read 的形状一致 [B, topk, d]
            q_s_expanded = q_s.unsqueeze(1).expand(-1, m_read.shape[1], -1)  # [B, topk, d]

            # 门控融合样本记忆和预测
            gate_s = torch.sigmoid(self.gate_proj(torch.cat([q_s_expanded, m_read], dim=-1)) + self.gate_bias)  # [B, topk, d]
            fuse_s = self.mem_fuse(torch.cat([q_s_expanded, m_read], dim=-1)) * self.mem_scale  # [B, topk, d]

            # 使用记忆库的结果进行加权融合
            final_with_memory = (1.0 + gate_s) * q_s_expanded + gate_s * fuse_s  # [B, topk, d]

            # [B, topk, d] -> [B, total_len, d]
            final_with_memory = rearrange(final_with_memory, 'B topk d -> B d topk')
            final_with_memory = self.topk_to_total_linear(final_with_memory)
            final_with_memory = rearrange(final_with_memory, 'B d total_len -> B total_len d')

            # ========================= 写入记忆库 =========================
            if self.training:
                write_mask = torch.ones(B, dtype=torch.bool, device=q_s.device)  # 所有样本都写入
                self.memory.write(
                    sample_ids[write_mask].detach(),
                    k_batch=F.normalize(q_s[write_mask].detach(), dim=-1),
                    v_batch=F.normalize(final_with_memory[write_mask].detach(), dim=-1)
                )
        else:
            final_with_memory = final  # 如果没有记忆库，直接用 final

        final_with_memory = final_with_memory + trend
        # ---------- 输出 ----------
        y = self.proj_out(final_with_memory)  # [B, total_len, 1]
        y = y[:, -self.pred_len:, :]  # [B, pred_len, 1]

        # ---------- RevIN inverse ----------
        if self.revin:
            mean, std = stats
            y = y * std[:, :1, :] + mean[:, :1, :]

        return y
