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
from modules.IFT_EncDec import ImplicitForecaster

class RetrievalResidualMemory(nn.Module):
    """
    检索残差记忆库：
    - key: 低维 query/key 向量 [B, key_dim]
    - value: 残差序列 [B, pred_len, 1]
    - 检索: cosine top-k -> softmax 加权 -> 残差期望
    - 写入: ring buffer，支持 sample_ids 做“近邻屏蔽”(可选)
    """
    def __init__(self, key_dim: int, pred_len: int, capacity: int = 8192, topk: int = 8):
        super().__init__()
        self.key_dim = key_dim
        self.pred_len = pred_len
        self.capacity = capacity
        self.topk = topk

        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, pred_len, 1))
        self.register_buffer("ids", torch.full((capacity,), -1, dtype=torch.long))

        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("size", torch.zeros((), dtype=torch.long))

        self.tau_raw = nn.Parameter(torch.tensor(0.0))  # softplus(tau)+1e-3

    @torch.no_grad()
    def add(self, key: torch.Tensor, residual: torch.Tensor, sample_ids: torch.Tensor = None):
        """
        key:      [B, key_dim]  (建议外部先 normalize)
        residual: [B, pred_len, 1]
        sample_ids(optional): [B]
        """
        B = key.size(0)
        assert key.shape == (B, self.key_dim)
        assert residual.shape == (B, self.pred_len, 1)

        for i in range(B):
            p = int(self.ptr.item())
            self.keys[p].copy_(key[i])
            self.values[p].copy_(residual[i])
            if sample_ids is not None:
                self.ids[p] = sample_ids[i].long()
            else:
                self.ids[p] = -1

            self.ptr.copy_(torch.tensor((p + 1) % self.capacity, device=self.ptr.device))
            self.size.copy_(torch.tensor(min(int(self.size.item()) + 1, self.capacity), device=self.size.device))

    def retrieve(self, query: torch.Tensor, sample_ids: torch.Tensor = None, block_same_id: bool = True):
        """
        query: [B, key_dim]
        return:
        residual_hat: [B, pred_len, 1]
        sims_pad:     [B, topk]   (固定维度，便于 QE)
        """
        B = query.size(0)
        n = int(self.size.item())
        if n == 0:
            resid0 = torch.zeros(B, self.pred_len, 1, device=query.device)
            sims0  = torch.full((B, self.topk), -1e9, device=query.device)
            return resid0, sims0

        k = min(self.topk, n)

        # buffers 已经在正确 device 上的话，不要 .to() 产生拷贝
        keys   = self.keys[:n].detach()      # [n, key_dim]  不让梯度进 buffer
        values = self.values[:n].detach()    # [n, pred_len, 1]
        ids    = self.ids[:n]                # [n]

        # cosine
        q  = F.normalize(query, dim=-1)
        kk = F.normalize(keys, dim=-1)
        sim = q @ kk.t()                     # [B, n]

        if block_same_id and (sample_ids is not None):
            sample_ids = sample_ids.to(sim.device).long()
            mask = (sample_ids.view(B, 1) == ids.view(1, n))
            sim = sim.masked_fill(mask, -1e9)

        sims_topk, idx_topk = torch.topk(sim, k=k, dim=-1)     # [B,k]

        # pad sims 到固定 [B, topk]
        if k < self.topk:
            pad = torch.full((B, self.topk - k), -1e9, device=sim.device)
            sims_pad = torch.cat([sims_topk, pad], dim=-1)
        else:
            sims_pad = sims_topk

        tau = F.softplus(self.tau_raw) + 1e-3
        w = torch.softmax(sims_topk / tau, dim=-1)             # [B,k]

        gathered = values[idx_topk]                             # [B,k,pred_len,1]
        residual_hat = (w.view(B, k, 1, 1) * gathered).sum(dim=1)

        return residual_hat, sims_pad


class QualityEstimatorLite(nn.Module):
    """
    输入：
      x_enc:  [B, seq_len, c_in]
      y_base: [B, pred_len, 1]
      sims:   [B, k]
    输出：
      beta:   [B, pred_len, 1]  (检索残差注入强度)
    """
    def __init__(self, seq_len: int, pred_len: int, c_in: int, k: int, d: int = 64):
        super().__init__()
        self.seq_proj = nn.Linear(seq_len, d)
        self.pred_proj = nn.Linear(pred_len, d)
        self.sims_proj = nn.Linear(k, d)

        self.loss_est = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.ReLU()
        )
        self.beta_head = nn.Sequential(
            nn.Linear(1 + k, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Sigmoid()
        )

    def forward(self, x_enc, y_base, sims):
        # x_enc -> [B,c_in,seq_len] -> [B,c_in,d] -> [B,d]
        x_feat = self.seq_proj(x_enc.permute(0, 2, 1)).mean(dim=1)

        # y_base -> [B,1,pred_len] -> [B,1,d] -> [B,d]
        y_feat = self.pred_proj(y_base.permute(0, 2, 1)).squeeze(1)

        # sims -> [B,d]
        s_feat = self.sims_proj(sims)

        loss_hat = self.loss_est(torch.cat([x_feat, y_feat, s_feat], dim=-1))  # [B,1]
        beta = self.beta_head(torch.cat([loss_hat, sims], dim=-1))             # [B,1]
        beta = beta.view(-1, 1, 1).expand(-1, y_base.size(1), 1)               # [B,pred_len,1]
        return loss_hat, beta

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
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.3):
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

class ThreeExpertPatchTransformer(nn.Module):
    """
    Three-expert Patch Transformer (Shared Backbone + Expert Heads + Optional TinyBottleneckMemory).

    Key design:
    - Shared backbone: run intra/inter patch transformer ONCE
    - Expert-specific: (optional) memory tail injection + expert head
    - Add one extra tail refinement layer (only on last pred_len tokens)
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
        c_in: int = 8
    ):
        super().__init__()
        self.config = config
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.d_model = d_model
        self.use_memory = use_memory
        self.mem_mode = getattr(config, "mem_mode", "none")
        self.momentum = getattr(config, "momentum", 0.05)
        self.use_decoding = getattr(config, "use_decoding", False)
        self.r = getattr(config, "r", 8)
        self.c_in = c_in
        self.num_heads = num_heads
        self.num_layers_intra_patch = num_layers_intra_patch
        self.num_layers_inter_patch = num_layers_inter_patch
        self.patch_len = patch_len
        self.win_size = win_size
        self.seq_weight = getattr(config, "seq_weight", 1.0)

        self.lambda_div = getattr(config, "lambda_div", 0.0)

        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len  # P

        # ---- Embedding ----
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.3)

        # ---- Prediction tokens (append to history) ----
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        # ---------- GMM weight branch (ww) ----------
        self.L_out0 = nn.Linear(1, d_model // 2)
        self.L_out1 = nn.Linear(1, d_model // 2)
        self.L_out2 = nn.Linear(1, d_model // 2)

        self.pos_embedding = PositionalEmbedding(d_model // 2, max_len=self.pred_len)

        self.attn0 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn1 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn4 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn5 = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.L_out10 = nn.Linear(d_model, 1)
        self.L_out11 = nn.Linear(d_model, 1)
        self.L_out12 = nn.Linear(d_model, 1)

        self.ln0 = nn.RMSNorm(d_model)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)
        
        
        # config 建议新增：
        # mem_mode="retrieval"
        # mem_size=8192
        # mem_topk=8
        # mem_key_dim=64
        # refine_d=64  (QualityEstimatorLite 用)

        if self.use_memory and self.mem_mode == "retrieval":
            self.mem_key_dim = getattr(config, "mem_key_dim", 64)
            self.mem_size = getattr(config, "mem_size", 8192)
            self.mem_topk = getattr(config, "mem_topk", 8)

            # 用共享骨干的历史 embedding 做 query/key（更稳，不依赖手工 x 切片）
            self.mem_key_proj = nn.Linear(self.d_model, self.mem_key_dim)

            self.memory = RetrievalResidualMemory(
                key_dim=self.mem_key_dim,
                pred_len=self.pred_len,
                capacity=self.mem_size,
                topk=self.mem_topk
            )

            self.qe = QualityEstimatorLite(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                c_in=self.c_in,
                k=self.mem_topk,
                d=getattr(config, "refine_d", 64)
            )


        # ---- Shared backbone blocks ----
        def _make_backbone():
            intra = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.3)
                for _ in range(self.num_layers_intra_patch)
            ])
            inter = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.3)
                for _ in range(self.num_layers_inter_patch)
            ])
            return intra, inter

        self.intra, self.inter = _make_backbone()
        self.post_norm = nn.RMSNorm(d_model)

        # ---- Temperature for ww softmax ----
        self.tau_raw = nn.Parameter(torch.tensor(0.0))  # tau = softplus + 1e-3

        # ---------- Tail refinement (one extra layer on prediction tail) ----------
        self.tail_refine = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # ---------- Heads (3 experts) ----------
        self.headA = nn.Linear(d_model, 1)
        self.headB = nn.Linear(d_model, 1)
        self.headC = nn.Linear(d_model, 1)

        # Optional decoding path
        self.enc_linear = nn.Linear(d_model, self.c_in)
        self.forecaster = ImplicitForecaster(self.config)

    # ====== ww builder: x -> ww ∈ [B, pred_len, 3] ======
    def _build_ww(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, c_in] (NOTE: your current slicing expects c_in>=8)
        Use x[:,:,5:8] (seq-level) and x[:,:,2:5] (point-level), fuse -> logits -> softmax.
        """
        B = x.size(0)
        relu = nn.ReLU()
        tanh = nn.Tanh()

        weight_seq = x[:, :, 5:8]            # [B, seq_len, 3]
        weight_pt = x[:, :, 2:5]             # [B, seq_len, 3]
        ww = weight_seq + self.seq_weight * weight_pt
        ww = ww[:, -self.pred_len:, :]       # [B, L, 3]

        # pos emb (kept compatible with your existing PositionalEmbedding usage)
        ww_emb = self.pos_embedding(ww).repeat(B, 1, 1)  # [B, L, d_model//2]

        # branch 0
        ww0 = ww[:, :, 0:1]
        ww0 = tanh(self.L_out0(ww0))                         # [B,L,d/2]
        ww0 = torch.cat([ww0, ww_emb], dim=-1)               # [B,L,d]
        z, _ = self.attn0(ww0, ww0, ww0)
        ww0 = self.ln0(ww0 + z)
        z, _ = self.attn3(ww0, ww0, ww0)
        ww0 = self.ln0(ww0 + z)
        ww0 = self.L_out10(relu(ww0))                        # [B,L,1]

        # branch 1
        ww1 = ww[:, :, 1:2]
        ww1 = tanh(self.L_out1(ww1))
        ww1 = torch.cat([ww1, ww_emb], dim=-1)
        z, _ = self.attn1(ww1, ww1, ww1)
        ww1 = self.ln1(ww1 + z)
        z, _ = self.attn4(ww1, ww1, ww1)
        ww1 = self.ln1(ww1 + z)
        ww1 = self.L_out11(relu(ww1))                        # [B,L,1]

        # branch 2
        ww2 = ww[:, :, 2:3]
        ww2 = tanh(self.L_out2(ww2))
        ww2 = torch.cat([ww2, ww_emb], dim=-1)
        z, _ = self.attn2(ww2, ww2, ww2)
        ww2 = self.ln2(ww2 + z)
        z, _ = self.attn5(ww2, ww2, ww2)
        ww2 = self.ln2(ww2 + z)
        ww2 = self.L_out12(relu(ww2))                        # [B,L,1]

        logits = torch.cat([ww0, ww1, ww2], dim=-1)          # [B,L,3]
        tau = F.softplus(self.tau_raw) + 1e-3
        ww = torch.softmax(logits / tau, dim=-1)
        return ww

    def _forward_backbone(
        self,
        x_emb: torch.Tensor,
        intra_blocks: nn.ModuleList,
        inter_blocks: nn.ModuleList,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        post_norm: nn.Module
    ) -> torch.Tensor:
        """
        Shared backbone forward.

        x_emb: [B, total_len, d]
        returns: [B, total_len, d]
        """
        B = x_emb.size(0)

        # 1) split to patches: [B, total_len, d] -> [B, P, pl, d]
        patches = rearrange(x_emb, "b (p pl) d -> b p pl d", p=self.num_patches, pl=self.patch_len)

        # 2) intra attention per patch (shared): merge (B,P) -> (B*P)
        patches_intra = rearrange(patches, "b p pl d -> (b p) pl d").contiguous()
        for block in intra_blocks:
            patches_intra = block(patches_intra, attn_mask=intra_mask)
        patches_intra = rearrange(
            patches_intra, "(b p) pl d -> b p pl d", b=B, p=self.num_patches
        ).contiguous()

        # flatten back
        intra_tokens = rearrange(patches_intra, "b p pl d -> b (p pl) d")

        # 3) inter attention across patches (shared): [B, P, pl, d] -> [B*pl, P, d]
        inter_patches = rearrange(patches_intra, "b p pl d -> (b pl) p d")
        for block in inter_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_mask)
        inter_tokens = rearrange(
            inter_patches, "(b pl) p d -> b (p pl) d", b=B, pl=self.patch_len
        )

        # 4) fuse + norm
        return post_norm(intra_tokens + inter_tokens)

    def forward(self, x: torch.Tensor, x_mark=None, y_true=None, sample_ids=None):
        """
        x: [B, seq_len, c_in]   (your ww slicing assumes c_in>=8)
        returns y: [B, pred_len, 1]
        """
        # ---- ww ----
        ww = self._build_ww(x)  # [B, L, 3]

        # ---- embed history + pred tokens ----
        x_emb_hist = self.embedding(x)  # [B, seq_len, d_model]
        B = x_emb_hist.size(0)
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)            # [B, total_len, d_model]

        # ---- masks ----
        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # ---- shared backbone (run once) ----
        final_shared = self._forward_backbone(
            x_emb, self.intra, self.inter, intra_mask, inter_mask, self.post_norm
        )  # [B, total_len, d_model]

        # expert "views" (no extra backbone)
        finalA = final_shared
        finalB = final_shared
        finalC = final_shared

        # ---- tail refine (one extra layer) ----
        def refine_tail_tokens(tokens):
            head = tokens[:, :-self.pred_len, :]
            tail = self.tail_refine(tokens[:, -self.pred_len:, :])
            return torch.cat([head, tail], dim=1)

        finalA = refine_tail_tokens(finalA)
        finalB = refine_tail_tokens(finalB)
        finalC = refine_tail_tokens(finalC)

        # ---- decoding or direct heads ----
        if self.use_decoding:
            finalA_ = self.enc_linear(finalA).permute(0, 2, 1)  # [B,c_in,total_len]
            finalB_ = self.enc_linear(finalB).permute(0, 2, 1)
            finalC_ = self.enc_linear(finalC).permute(0, 2, 1)

            yA = self.forecaster(finalA_, x)[:, :self.pred_len, :]  # [B,L,1]
            yB = self.forecaster(finalB_, x)[:, :self.pred_len, :]
            yC = self.forecaster(finalC_, x)[:, :self.pred_len, :]
        else:
            yA = self.headA(finalA[:, -self.pred_len:, :])          # [B,L,1]
            yB = self.headB(finalB[:, -self.pred_len:, :])
            yC = self.headC(finalC[:, -self.pred_len:, :])

        # ---- weighted fusion ----
        w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]
        
        y_base = w0 * yA + w1 * yB + w2 * yC   # [B,pred_len,1]

        if self.use_memory and self.mem_mode == "retrieval":
            # query 用 backbone 的历史表示更稳
            q = self.mem_key_proj(x_emb_hist.mean(dim=1))   # [B,key_dim]
            q = F.normalize(q, dim=-1)

            # 取目标列（必须）
            if y_true is not None:
                y_t = y_true[:, :self.pred_len, 0:1]        # [B,72,1] 这里的 0:1 改成你的目标列
            else:
                y_t = None

            # 检索（空库会返回 0）
            resid_hat, sims = self.memory.retrieve(q, sample_ids=sample_ids, block_same_id=True)

            loss_hat, beta = self.qe(x, y_base.detach(), sims)     # 这里 y_base 可 detach 稳一点
            y = y_base + beta * resid_hat
            
            # 写入（训练时且有标签）
            if self.training and (y_t is not None):
                residual = (y_t - y_base).detach()
                self.memory.add(q.detach(), residual, sample_ids=sample_ids)
        else:
            y = y_base

        
        return y