#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer with Memory

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from layers.embedding import DataEmbedding
from layers.embedding import DataEmbedding
from layers.embedding import PositionalEmbedding


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
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
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


# ========= 主模型：三专家 Patch Transformer + GMM 权重门控 =========
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
        c_in: int = 8                  # 输入通道数
    ):
        super().__init__()
        self.config = config
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.d_model = d_model
        self.use_memory = use_memory
        self.mem_mode = config.mem_mode
        self.momentum = config.momentum
        self.r = config.r
        self.num_heads = num_heads
        self.num_layers_intra_patch = num_layers_intra_patch
        self.num_layers_inter_patch = num_layers_inter_patch
        self.patch_len = patch_len
        self.win_size = win_size
        self.seq_weight = config.seq_weight  # 融合点级与序列级 GMM 权重的系数

        self.lambda_div = getattr(config, "lambda_div", 0.0)  # 专家多样性正则系数
        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len  # P
       
        # DataEmbedding（形状: [B,L,c_in] -> [B,L,d_model]）
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.2)

        # 预测 tokens（替代 predict_linear 的外推） ===
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

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
        self.ln0 = nn.RMSNorm(d_model)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

        # ---------- 三套独立 Transformer 主干（A/B/C） ----------
        def _make_backbone():
            # Intra：每个 patch 一组堆叠的块
            intra = nn.ModuleList([
                nn.ModuleList([
                    TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.2)
                    for _ in range(self.num_layers_intra_patch)
                ])
                for _ in range(self.num_patches)
            ])
            # Inter：跨 patch 的块
            inter = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.2)
                for _ in range(self.num_layers_inter_patch)
            ])
            return intra, inter
        
        self.intraA, self.interA = _make_backbone()
        self.intraB, self.interB = _make_backbone()
        self.intraC, self.interC = _make_backbone()

       
        # 主干输出后规范化
        self.post_normA = nn.RMSNorm(d_model)
        self.post_normB = nn.RMSNorm(d_model)
        self.post_normC = nn.RMSNorm(d_model)

        # ---------- GMM 温度（可学习，>0） ----------
        self.tau_raw = nn.Parameter(torch.tensor(0.0))  # τ = softplus(tau_raw)+1e-3

        # ---------- 三个专家头 ----------
        self.headA = nn.Linear(d_model, 1)
        self.headB = nn.Linear(d_model, 1)
        self.headC = nn.Linear(d_model, 1)

    # ====== GMM 权重分支：把 x -> ww ∈ [B, pred_len, 3] ======
    def _build_ww(self, x):
        """
        取 x 的 [5:8] 作为序列级、[2:5] 作为点级，线性融合 -> 三通道；经过三路注意力 + MLP 降到标量 logit；
        softmax 得到逐步三专家权重。
        """
        B = x.size(0)
        relu = nn.ReLU()
        tanh = nn.Tanh()

        # 融合得到三通道权重特征（与输出步数对齐，仅取最后 pred_len 步）
        weight_seq = x[:, :, 5:8]  # [B, total_len, 3]
        weight_pt  = x[:, :, 2:5]  # [B, total_len, 3]
        ww = weight_seq + self.seq_weight * weight_pt
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

        logits = torch.cat([ww0, ww1, ww2], dim=-1)    # [B, L, 3]
        tau = F.softplus(self.tau_raw) + 1e-3
        ww = torch.softmax(logits / tau, dim=-1)       # [B, L, 3]
        return ww
    
    def _forward_backbone(self, x_emb, intra_blocks, inter_blocks, intra_mask, inter_mask, post_norm):
        """
        x_emb: [B, total_len, d]
        返回:
            final: [B, total_len, d]
        这里实现的是：先对每个 patch 做 Intra，再在 patch 之间做 Inter（串联设计）
        """
        B = x_emb.size(0)

        # 1) 划分 patch
        patches = rearrange(
            x_emb, 'b (p pl) d -> b p pl d',
            p=self.num_patches, pl=self.patch_len
        )  # [B, P, pl, d]

        # 2) Intra: 对每个 patch 走自己的一组块，但不 in-place 写回原 view
        updated_patches = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]  # [B, pl, d]
            for block in intra_blocks[p]:
                out = block(out, attn_mask=intra_mask)  # [B, pl, d]
            updated_patches.append(out.unsqueeze(1))    # [B, 1, pl, d]

        # 拼成新的 patch 张量（已经是 Intra 之后的结果）
        patches_intra = torch.cat(updated_patches, dim=1)  # [B, P, pl, d]

        # 展平成 token 序列，作为 Intra 分支输出
        intra_tokens = rearrange(patches_intra, 'b p pl d -> b (p pl) d')  # [B, total_len, d]

        # 3) Inter: 在 patch 维做自注意力（用 Intra 之后的 patches_intra）
        inter_patches = rearrange(patches_intra, 'B P pl d -> (B pl) P d')  # [B*pl, P, d]
        for block in inter_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_mask)      # [B*pl, P, d]
        inter_tokens = rearrange(
            inter_patches, '(B pl) P d -> B (P pl) d',
            B=B, pl=self.patch_len
        )  # [B, total_len, d]

        # 4) Intra + Inter 融合，再做层归一化
        final = intra_tokens + inter_tokens         # [B, total_len, d]
        final = post_norm(final)
        return final



    def forward(self, x, x_mark=None, y_true=None, sample_ids=None):
        
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # ---------- GMM 权重分支（得到 ww ∈ [B, pred_len, 3]） ----------
        ww = self._build_ww(x)  # [B, L, 3], L=pred_len

        # === 用预测 tokens 拼成 total_len ===
        x_emb_hist = self.embedding(x)                                # [B, seq_len, d]
        B = x_emb_hist.size(0)                                        # 取出Batch维度
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)            # [B, total_len, d]

        # ---------- 构造掩码 ----------
        intra_mask = generate_causal_window_mask(
            self.patch_len, self.win_size, x_emb.device, x_emb.dtype
        )  # [pl, pl] 
        inter_mask = generate_causal_window_mask(
            self.num_patches, self.win_size, x_emb.device, x_emb.dtype
        )  # [P, P]

        # 用 ww 做一个样本级的路由系数（平均到时间维）
        mix = ww.mean(dim=1)           # [B, 3]
        g0 = mix[:, 0:1].unsqueeze(-1) # [B, 1, 1]
        g1 = mix[:, 1:2].unsqueeze(-1)
        g2 = mix[:, 2:3].unsqueeze(-1)

        # 这里用 (1 + g) 而不是直接乘 g，避免某一路完全被压成 0
        x_embA = x_emb * (1.0 + g0)
        x_embB = x_emb * (1.0 + g1)
        x_embC = x_emb * (1.0 + g2)
        
        # ---------- 三个专家主干 ----------
        finalA = self._forward_backbone(x_embA, self.intraA, self.interA, intra_mask, inter_mask, self.post_normA)  # [B, total_len, d]
        finalB = self._forward_backbone(x_embB, self.intraB, self.interB, intra_mask, inter_mask, self.post_normB)
        finalC = self._forward_backbone(x_embC, self.intraC, self.interC, intra_mask, inter_mask, self.post_normC)

        # ---------- 三个专家头，切出预测区间 ----------
        yA = self.headA(finalA)[:, -self.pred_len:, :]  # [B, L, 1]
        yB = self.headB(finalB)[:, -self.pred_len:, :]  # [B, L, 1]
        yC = self.headC(finalC)[:, -self.pred_len:, :]  # [B, L, 1]

        # ---------- ww 逐步加权融合 ----------
        w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]  # [B, L, 1] x 3
        y = w0 * yA + w1 * yB + w2 * yC                       # [B, L, 1]
        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return y