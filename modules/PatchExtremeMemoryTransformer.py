#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer with Sample-level Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.embedding import DataEmbedding
from layers.embedding import PositionalEmbedding

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
        self.seq_w = 0.3
        self.top_k = 8
        self.kernel_size = 25

        # --------- Embedding & 预投影到 total_len ----------
        # self.embedding = DataEmbedding(c_in=2, d_model=d_model, dropout=0.1)
        self.embedding = DataEmbedding(c_in=8, d_model=d_model, dropout=0.5)
        self.pos_embedding = PositionalEmbedding(self.pred_len)
        self.predict_linear = nn.Linear(seq_len, self.total_len)  # [B,d,seq] -> [B,d,total]
        self.topk_to_total_linear = nn.Linear(mem_topk, self.total_len)
        # --------- 周期趋势分解 ----------
        self.dft = DFT(top_k=self.top_k)
        self.trend = moving_avg(kernel_size=self.kernel_size, stride=1)


        # 三个分支的升维层（从 1 -> atten_dim/2）
        self.L_out0 = nn.Linear(1, int(d_model / 2))
        self.L_out1 = nn.Linear(1, int(d_model / 2))
        self.L_out2 = nn.Linear(1, int(d_model / 2))

        self.attn0 = nn.MultiheadAttention(d_model, 4)
        self.attn1 = nn.MultiheadAttention(d_model, 4)
        self.attn2 = nn.MultiheadAttention(d_model, 4)
        self.attn3 = nn.MultiheadAttention(d_model, 4)
        self.attn4 = nn.MultiheadAttention(d_model, 4)
        self.attn5 = nn.MultiheadAttention(d_model, 4)

        # 线性层：将标量分支先抬升到 atten_dim/2，拼接位置编码后得到 atten_dim
        # 再用线性层降回标量，作为该分支的“logit”
        self.L_out00 = nn.Linear(1, d_model)     # （未使用：保留）
        self.L_out10 = nn.Linear(d_model, 1)
        self.L_out01 = nn.Linear(1, d_model)     # （未使用：保留）
        self.L_out11 = nn.Linear(d_model, 1)
        self.L_out02 = nn.Linear(1, d_model)     # （未使用：保留）
        self.L_out12 = nn.Linear(d_model, 1)

         # 批归一化：用 output_len 作为“通道数”，等价于沿时间维做归一化（非常见用法）
        # 输入形状 [B, T, C] 会被当作 [N, C, L]，即 T 作为通道维进行 BN
        self.bn = nn.BatchNorm1d(self.pred_len)

        # =====================================================
        # 分支 A：Intra-branch（patch 内）
        # =====================================================
        self.intra_patch_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.5)
                for _ in range(self.num_layers_intra_patch)
            ])
            for _ in range(self.num_patches)
        ])

        # =====================================================
        # 分支 B：Inter-branch（patch 间）
        # =====================================================
        self.inter_patch_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.5)
            for _ in range(self.num_layers_inter_patch)
        ])

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

    
    def forward(self, x, x_mark=None, sample_ids=None):
        # x是原始序列  x_mark是GMM辅助序列
        B, _, _ = x.shape
        
        # RevIN
        if self.revin:
            mean = x.mean(1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / std
            stats = (mean, std)
        
        # # 处理GMM的权重生成
        # weight0 = x_mark[:, :, 5:8] # 序列级权重
        # weight1 = x_mark[:, :, 2:5] # 点级权重
        # ww = torch.add(weight0, weight1 * self.seq_w) # 融合得到3通道权重特征
        # ww = ww[:, -1 * self.pred_len:, :] # 与输出的时间步对齐，方便加权融合
        # ww_emb = self.pos_embedding(ww) # 对GMM权重做embedding
        # ww_emb = ww_emb.repeat(weight0.size(0), 1, 1)
        
        # weight0 = ww[:, :, 0:1]                # [B, L, 1]
        # weight0 = nn.Tanh(self.L_out0(weight0))       # [B, L, atten_dim/2] 升维
        # weight0 = torch.cat((weight0, ww_emb), dim=-1) # [B, L, atten_dim] 拼接位置编码
        
        # # 自注意力（层 1）
        # ww00, _ = self.attn0(ww0, ww0, ww0)
        # # 残差 + “时间维 BN”（非常见但可行）
        # ww0 = self.bn(ww0 + ww00)
        # # 自注意力（层 2）
        # ww00, _ = self.attn3(ww0, ww0, ww0)
        # ww0 = self.bn(ww0 + ww00)
        # # 降维为标量 logit
        # ww0 = self.L_out10(nn.ReLU(ww0))      # [B, L, 1]


        # # 分支 1：同理处理第二个通道
        # ww1 = ww[:, :, 1:2]
        # ww1 = nn.Tanh(self.L_out1(ww1))
        # ww1 = torch.cat((ww1, ww_emb), dim=2)
        # ww11, _ = self.attn1(ww1, ww1, ww1)
        # ww1 = self.bn(ww1 + ww11)
        # ww11, _ = self.attn4(ww1, ww1, ww1)
        # ww1 = self.bn(ww1 + ww11)
        # ww1 = self.L_out11(nn.ReLU(ww1))      # [B, L, 1]

        # # 分支 2：同理处理第三个通道
        # ww2 = ww[:, :, 2:3]
        # ww2 = self.L_out2(ww2)
        # ww2 = torch.cat((ww2, ww_emb), dim=2)
        # ww22, _ = self.attn2(ww2, ww2, ww2)
        # ww2 = self.bn(ww2 + ww22)
        # ww22, _ = self.attn5(ww2, ww2, ww2)
        # ww2 = self.bn(ww2 + ww22)
        # ww2 = self.L_out12(nn.ReLU(ww2))      # [B, L, 1]

        #  # 合并三个分支的标量 logit：[B, L, 3]，随后做 softmax 得到 3 专家的概率权重
        # ww = torch.cat((ww0, ww1), dim=-1)  # [B, L, 2]
        # ww = torch.cat((ww, ww2), dim=-1)   # [B, L, 3]
        # ww = nn.Softmax(ww, dim=-1)                         # softmax 按 dim=2，保证同一时间步三权重和为 1

        # ---------- Embedding & 预投影 ----------
        x_emb = self.embedding(x)             # [B, L, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')     # [B, d, L]
        x_emb = self.predict_linear(x_emb)             # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')     # [B, total_len, d]

        
        # ---------- Patch Division ----------
        patches = rearrange(x_emb, 'b (p pl) d -> b p pl d', p=self.num_patches, pl=self.patch_len)

        # intra: [patch_len, patch_len]
        intra_patch_mask = generate_causal_window_mask(
            self.patch_len, self.win_size, patches.device, patches.dtype
        )

        # inter: [num_patches, num_patches]
        inter_patch_mask = generate_causal_window_mask(
            self.num_patches, self.win_size, patches.device, patches.dtype
        )

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

        final_with_memory = final_with_memory
        # ---------- 输出 ----------
        y = self.proj_out(final_with_memory)  # [B, total_len, 1]
        y = y[:, -self.pred_len:, :]  # [B, pred_len, 1]

        # ---------- RevIN inverse ----------
        if self.revin:
            mean, std = stats
            y = y * std[:, :1, :] + mean[:, :1, :]

        return y

# ========= 主模型：三专家 Patch Transformer + GMM 权重门控 =========
class ThreeExpertPatchTransformer(nn.Module):
    """
    方案B：三套完全独立的 Transformer 主干（专家 A/B/C） + GMM 权重分支 ww（3通道）做逐步加权融合。
    - 输入：
        x:       [B, seq_len, 2]             （原始序列的两个通道；按你工程可调整 c_in）
        x_mark:  [B, total_len, C_mark>=8]   （包含 GMM 的辅助特征，至少能切到 [:, :, 2:5] 和 [:, :, 5:8]）
        sample_ids: [B] or [B, ...]          （记忆库用到）
    - 输出：
        y: [B, pred_len, 1]
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
        config=None,
        mem_size: int = 4096,
        mem_topk: int = 16,
        mem_tau: float = 0.5,
        mem_momentum: float = 0.2,
        c_in: int = 8,                   # 输入通道数（你的 DataEmbedding 缺省是 2）
    ):
        super().__init__()
        self.config = config
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.d_model = d_model
        self.use_memory = use_memory
        self.num_heads = num_heads
        self.num_layers_intra_patch = num_layers_intra_patch
        self.num_layers_inter_patch = num_layers_inter_patch
        self.patch_len = patch_len
        self.win_size = win_size
        self.seq_w = 0.3   # 融合点级与序列级 GMM 权重的系数
        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len  # P

        # ---------- 序列嵌入与长度映射 ----------
        # 说明：这里依赖你工程里的 DataEmbedding（形状: [B,L,c_in] -> [B,L,d_model]）
        from layers.embedding import DataEmbedding  # 若路径不同请修改
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.5)

        # 把 [B, d, seq_len] 投影到 [B, d, total_len]，以便拼上 pred_len 的 token（一次性多步）
        self.predict_linear = nn.Linear(seq_len, self.total_len)

        # ---------- GMM 权重分支（ww） ----------
        # 三个“标量→升维”的线性层：1 -> d_model//2
        self.L_out0 = nn.Linear(1, d_model // 2)
        self.L_out1 = nn.Linear(1, d_model // 2)
        self.L_out2 = nn.Linear(1, d_model // 2)

        # 位置编码（用于 ww 分支），维度对齐 d_model//2
        self.pos_embedding = PositionalEmbedding(d_model // 2, max_len=self.pred_len)

        # 六个自注意力（每路两层），batch_first=True 以使用 [B, L, C]
        self.attn0 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn4 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn5 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # 降维为标量 logit
        self.L_out10 = nn.Linear(d_model, 1)
        self.L_out11 = nn.Linear(d_model, 1)
        self.L_out12 = nn.Linear(d_model, 1)

        # 更稳的归一化：LayerNorm（替代把时间当通道的 BN）
        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


        self.prefiltA = nn.Conv1d(d_model, d_model, kernel_size=51, padding=25, groups=d_model)
        self.prefiltB = nn.Conv1d(d_model, d_model, kernel_size=11, padding=5,  groups=d_model)
        self.prefiltC = nn.Conv1d(d_model, d_model, kernel_size=3,  padding=1,  groups=d_model)

        # ---------- 三套独立 Transformer 主干（A/B/C） ----------
        def _make_backbone():
            # Intra：每个 patch 一组堆叠的块
            intra = nn.ModuleList([
                nn.ModuleList([
                    TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.5)
                    for _ in range(self.num_layers_intra_patch)
                ])
                for _ in range(self.num_patches)
            ])
            # Inter：跨 patch 的块
            inter = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.5)
                for _ in range(self.num_layers_inter_patch)
            ])
            return intra, inter

        self.intraA, self.interA = _make_backbone()
        self.intraB, self.interB = _make_backbone()
        self.intraC, self.interC = _make_backbone()

        # 主干输出后规范化
        self.post_normA = nn.LayerNorm(d_model)
        self.post_normB = nn.LayerNorm(d_model)
        self.post_normC = nn.LayerNorm(d_model)

        # ---------- 记忆库（每个专家一套，可关掉） ----------
        if self.use_memory:
            from layers.memory import SampleMemory  # 若路径不同请修改
            self.memoryA = SampleMemory(d_model=d_model, mem_size=mem_size, topk=mem_topk,
                                        temperature=mem_tau, ema_momentum=mem_momentum)
            self.memoryB = SampleMemory(d_model=d_model, mem_size=mem_size, topk=mem_topk,
                                        temperature=mem_tau, ema_momentum=mem_momentum)
            self.memoryC = SampleMemory(d_model=d_model, mem_size=mem_size, topk=mem_topk,
                                        temperature=mem_tau, ema_momentum=mem_momentum)

            def _mem_fuser():
                return nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model)
                )
            self.mem_fuseA = _mem_fuser()
            self.mem_fuseB = _mem_fuser()
            self.mem_fuseC = _mem_fuser()

            self.gate_projA = nn.Linear(2 * d_model, d_model)
            self.gate_projB = nn.Linear(2 * d_model, d_model)
            self.gate_projC = nn.Linear(2 * d_model, d_model)

            self.mem_scaleA = nn.Parameter(torch.tensor(1.0))
            self.mem_scaleB = nn.Parameter(torch.tensor(1.0))
            self.mem_scaleC = nn.Parameter(torch.tensor(1.0))

            self.gate_biasA = nn.Parameter(torch.zeros(1))
            self.gate_biasB = nn.Parameter(torch.zeros(1))
            self.gate_biasC = nn.Parameter(torch.zeros(1))

            # 将 topk 维映射回 total_len（作用在最后一维）
            self.topk_to_total_linear = nn.Linear(mem_topk, self.total_len)

        # ---------- 三个专家头 ----------
        self.headA = nn.Linear(d_model, 1)
        self.headB = nn.Linear(d_model, 1)
        self.headC = nn.Linear(d_model, 1)

    # ====== GMM 权重分支：把 x_mark -> ww ∈ [B, pred_len, 3] ======
    def _build_ww(self, x):
        """
        取 x_mark 的 [5:8] 作为序列级、[2:5] 作为点级，线性融合 -> 三通道；经过三路注意力 + MLP 降到标量 logit；
        softmax 得到逐步三专家权重。
        """
        B = x.size(0)
        relu = nn.ReLU()
        tanh = nn.Tanh()

        # 融合得到三通道权重特征（与输出步数对齐，仅取最后 pred_len 步）
        weight_seq = x[:, :, 5:8]  # [B, total_len, 3]
        weight_pt  = x[:, :, 2:5]  # [B, total_len, 3]
        ww = weight_seq + self.seq_w * weight_pt
        ww = ww[:, -self.pred_len:, :]  # [B, L, 3]  L = pred_len

        # 位置编码（d_model//2 维）
        ww_emb = self.pos_embedding(ww).repeat(B, 1, 1)  # [B, L, d_model//2]

        # 分支0
        ww0 = ww[:, :, 0:1]                       # [B, L, 1]
        ww0 = tanh(self.L_out0(ww0))              # [B, L, d_model//2]
        ww0 = torch.cat([ww0, ww_emb], dim=-1)    # [B, L, d_model]
        z, _ = self.attn0(ww0, ww0, ww0)
        ww0 = self.ln0(ww0 + z)
        z, _ = self.attn3(ww0, ww0, ww0)
        ww0 = self.ln0(ww0 + z)
        ww0 = self.L_out10(relu(ww0))             # [B, L, 1]

        # 分支1
        ww1 = ww[:, :, 1:2]
        ww1 = tanh(self.L_out1(ww1))
        ww1 = torch.cat([ww1, ww_emb], dim=-1)
        z, _ = self.attn1(ww1, ww1, ww1)
        ww1 = self.ln1(ww1 + z)
        z, _ = self.attn4(ww1, ww1, ww1)
        ww1 = self.ln1(ww1 + z)
        ww1 = self.L_out11(relu(ww1))             # [B, L, 1]

        # 分支2
        ww2 = ww[:, :, 2:3]
        ww2 = tanh(self.L_out2(ww2))
        ww2 = torch.cat([ww2, ww_emb], dim=-1)
        z, _ = self.attn2(ww2, ww2, ww2)
        ww2 = self.ln2(ww2 + z)
        z, _ = self.attn5(ww2, ww2, ww2)
        ww2 = self.ln2(ww2 + z)
        ww2 = self.L_out12(relu(ww2))             # [B, L, 1]

        # 拼接 -> 概率
        ww = torch.cat([ww0, ww1, ww2], dim=-1)   # [B, L, 3]
        ww = torch.softmax(ww, dim=-1)            # [B, L, 3]
        return ww

    # ====== 单个专家主干前向（intra + inter + norm）======
    def _forward_backbone(self, x_emb, intra_blocks, inter_blocks, intra_mask, inter_mask, post_norm):
        """
        x_emb:      [B, total_len, d]
        返回 final:  [B, total_len, d]
        """
        B = x_emb.size(0)
        # Patch 划分
        patches = rearrange(x_emb, 'b (p pl) d -> b p pl d', p=self.num_patches, pl=self.patch_len)

        # Intra：每个 patch 内的局部注意
        outs_intra = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]  # [B, pl, d]
            for block in intra_blocks[p]:
                out = block(out, attn_mask=intra_mask)  # 你的 TransformerBlock 要支持 attn_mask
            outs_intra.append(out)
        intra_tokens = torch.cat(outs_intra, dim=1)  # [B, P*pl, d] == [B, total_len, d]

        # Inter：跨 patch 的全局关系（把 pl 合并进 batch 维，便于共享 inter_mask）
        inter_patches = rearrange(patches, 'B P pl d -> (B pl) P d')
        for block in inter_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_mask)
        inter_tokens = rearrange(inter_patches, '(B pl) P d -> B (P pl) d', B=B, pl=self.patch_len)  # [B, total_len, d]

        final = intra_tokens + inter_tokens
        final = post_norm(final)
        return final  # [B, total_len, d]

    # ====== 记忆库读取/融合/映射回时间轴 ======
    def _apply_memory(self, final, memory, mem_fuse, gate_proj, mem_scale, gate_bias, sample_ids):
        """
        final: [B, total_len, d]（取其聚合向量 q_s 作为 key/query）
        返回：final_with_memory: [B, total_len, d]
        """
        B, L, D = final.shape
        q_s = final.sum(dim=1)                        # [B, d]
        m_read, _, _ = memory.read(sample_ids, q_s)   # m_read: [B, topk, d]

        # 对齐 gate 维度
        q_exp = q_s.unsqueeze(1).expand(-1, m_read.shape[1], -1)  # [B, topk, d]
        gate = torch.sigmoid(gate_proj(torch.cat([q_exp, m_read], dim=-1)) + gate_bias)  # [B, topk, d]
        fuse = mem_fuse(torch.cat([q_exp, m_read], dim=-1)) * mem_scale                  # [B, topk, d]

        # 记忆融合（topk 维度）
        fused = (1.0 + gate) * q_exp + gate * fuse          # [B, topk, d]

        # 把 topk 映射回 total_len： [B, topk, d] -> [B, d, topk] -> Linear(topk->total_len) -> [B, d, total_len] -> [B, total_len, d]
        fused = rearrange(fused, 'B K d -> B d K')
        fused = self.topk_to_total_linear(fused)             # 线性作用在最后一维
        fused = rearrange(fused, 'B d L -> B L d')

        # （可选）再与原 final 相加形成残差；若与你之前实现一致，可以直接返回 fused
        # return fused
        return final + fused

    def forward(self, x, x_mark=None, sample_ids=None):

        
        """
        x:       [B, seq_len, c_in]
        x_mark:  [B, total_len, C_mark]  —— 注意：本模型假定 x_mark 的时间长度是 total_len（或可至少切到最后 pred_len）
        """
        B = x.size(0)

        # # ---------- RevIN ----------
        # if self.revin:
        #     mean = x.mean(dim=1, keepdim=True).detach()  # [B,1,c]
        #     x_norm = x - mean
        #     std = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x = x_norm / std
        #     stats = (mean, std)

        # ---------- GMM 权重分支（得到 ww ∈ [B, pred_len, 3]） ----------
        assert x_mark is not None, "x_mark (包含 GMM 辅助特征) 不能为空"
        ww = self._build_ww(x)  # [B, L, 3], L=pred_len

        # ---------- 序列嵌入 & 映射到 total_len ----------
        x_emb = self.embedding(x)                      # [B, seq_len, d]
        x_emb = rearrange(x_emb, 'b l d -> b d l')     # [B, d, seq_len]
        x_emb = self.predict_linear(x_emb)             # [B, d, total_len]
        x_emb = rearrange(x_emb, 'b d l -> b l d')     # [B, total_len, d]

        # ---------- 构造掩码（来自你工程的函数） ----------
        
        intra_mask = generate_causal_window_mask(
            self.patch_len, self.win_size, x_emb.device, x_emb.dtype
        )  # [pl, pl] 或符合你 TransformerBlock 的期望
        inter_mask = generate_causal_window_mask(
            self.num_patches, self.win_size, x_emb.device, x_emb.dtype
        )  # [P, P]

        x_ch = rearrange(x_emb, 'b l d -> b d l')
        xA = rearrange(self.prefiltA(x_ch), 'b d l -> b l d')
        xB = rearrange(self.prefiltB(x_ch), 'b d l -> b l d')
        xC = rearrange(self.prefiltC(x_ch), 'b d l -> b l d')
        

        # ---------- 三个专家主干 ----------
        finalA = self._forward_backbone(xA, self.intraA, self.interA, intra_mask, inter_mask, self.post_normA)  # [B, total_len, d]
        finalB = self._forward_backbone(xB, self.intraB, self.interB, intra_mask, inter_mask, self.post_normB)
        finalC = self._forward_backbone(xC, self.intraC, self.interC, intra_mask, inter_mask, self.post_normC)

        # ---------- 记忆库（可选） ----------
        if self.use_memory:
            assert sample_ids is not None, "use_memory=True 时需要 sample_ids"
            finalA = self._apply_memory(finalA, self.memoryA, self.mem_fuseA, self.gate_projA, self.mem_scaleA, self.gate_biasA, sample_ids)
            finalB = self._apply_memory(finalB, self.memoryB, self.mem_fuseB, self.gate_projB, self.mem_scaleB, self.gate_biasB, sample_ids)
            finalC = self._apply_memory(finalC, self.memoryC, self.mem_fuseC, self.gate_projC, self.mem_scaleC, self.gate_biasC, sample_ids)

            # 写入（训练阶段）
            if self.training:
                write_mask = torch.ones(B, dtype=torch.bool, device=x_emb.device)
                # 使用规范化的 key/value
                kA = F.normalize(finalA.sum(dim=1).detach()[write_mask], dim=-1)  # [B,d]
                vA = F.normalize(finalA.detach()[write_mask], dim=-1)             # [B,L,d]
                self.memoryA.write(sample_ids[write_mask].detach(), k_batch=kA, v_batch=vA)

                kB = F.normalize(finalB.sum(dim=1).detach()[write_mask], dim=-1)
                vB = F.normalize(finalB.detach()[write_mask], dim=-1)
                self.memoryB.write(sample_ids[write_mask].detach(), k_batch=kB, v_batch=vB)

                kC = F.normalize(finalC.sum(dim=1).detach()[write_mask], dim=-1)
                vC = F.normalize(finalC.detach()[write_mask], dim=-1)
                self.memoryC.write(sample_ids[write_mask].detach(), k_batch=kC, v_batch=vC)

        # ---------- 三个专家头，切出预测区间 ----------
        yA = self.headA(finalA)[:, -self.pred_len:, :]  # [B, L, 1]
        yB = self.headB(finalB)[:, -self.pred_len:, :]  # [B, L, 1]
        yC = self.headC(finalC)[:, -self.pred_len:, :]  # [B, L, 1]

        # ---------- ww 逐步加权融合 ----------
        w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]  # [B, L, 1] x 3
        y = w0 * yA + w1 * yB + w2 * yC                       # [B, L, 1]

        # # ---------- RevIN 逆变换 ----------
        # if self.revin:
        #     mean, std = stats              # mean/std: [B,1,c]
        #     # 若你只预测目标通道（1个），这里默认用第一个通道的统计量；可按需改为目标通道的统计
        #     y = y * std[:, :1, :1] + mean[:, :1, :1]

        return y