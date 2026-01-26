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

# =========================================================
# Turning-Point detection + MemoryBank (for normalized 1st-diff)
# =========================================================

def _turning_score_from_diff(diff_1d: torch.Tensor,
                             eps: float = 0.05,
                             min_abs: float = 0.6,
                             min_jump: float = 0.6,
                             region_start: int = 0,
                             region_end: int = None):
    """
    diff_1d: [B, T]  (normalized 1st-difference)
    Returns:
      has_tp: [B] bool
      score:  [B] float (max turning score in region)
    Turning defined by sign flip with hysteresis + amplitude/jump thresholds.
    """
    B, T = diff_1d.shape
    if region_end is None:
        region_end = T - 1
    region_end = min(region_end, T - 1)

    # sign with dead-zone to suppress tiny oscillations near 0
    s = torch.sign(diff_1d)
    s = torch.where(diff_1d.abs() < eps, torch.zeros_like(s), s)

    # flip between t-1 and t  -> located at t
    flip = (s[:, 1:] * s[:, :-1] < 0)  # [B, T-1]

    d0 = diff_1d[:, :-1]  # [B, T-1]
    d1 = diff_1d[:, 1:]   # [B, T-1]
    amp_ok = (d0.abs() > min_abs) & (d1.abs() > min_abs)
    jump_ok = ((d1 - d0).abs() > min_jump)

    valid = flip & amp_ok & jump_ok  # [B, T-1]

    # region mask on "t" (i.e., transition index t in [1..T-1])
    # valid is indexed by t-1, corresponds to t in [1..T-1]
    t = torch.arange(1, T, device=diff_1d.device)  # [T-1]
    region = (t >= region_start) & (t <= region_end)
    valid = valid & region.view(1, -1)

    # turning score: stronger flip + larger jump => higher
    score_each = (d0.abs() + d1.abs() + (d1 - d0).abs())  # [B, T-1]
    score_each = torch.where(valid, score_each, torch.zeros_like(score_each))

    score, _ = score_each.max(dim=1)  # [B]
    has_tp = score > 0
    return has_tp, score


class TurningPointKeyEncoder(nn.Module):
    """
    Encode last key_len window of (normalized) 1st-diff target channel (+ optional extra channels)
    into a compact key vector for retrieval.
    """
    def __init__(self, in_ch: int, key_dim: int = 64, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Linear(4 * in_ch, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, key_dim),
        )

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        """
        x_win: [B, key_len, in_ch]
        Return: key [B, key_dim]
        """
        # statistics on window (robust for normalized diff)
        mean = x_win.mean(dim=1)
        std = x_win.std(dim=1, unbiased=False)
        max_abs = x_win.abs().amax(dim=1)
        last = x_win[:, -1, :]
        feat = torch.cat([mean, std, max_abs, last], dim=-1)  # [B, 4*in_ch]
        key = self.net(feat)
        key = F.normalize(key, dim=-1)
        return key


class TurningPointMemoryBank(nn.Module):
    """
    Ring-buffer memory for turning-point patterns.
      keys:   [N, key_dim]
      values: [N, pred_len, 1]  (store target diff trajectory)
    """
    def __init__(self, mem_size: int, key_dim: int, pred_len: int, topk: int = 8):
        super().__init__()
        self.mem_size = int(mem_size)
        self.key_dim = int(key_dim)
        self.pred_len = int(pred_len)
        self.topk = int(topk)

        self.register_buffer("keys", torch.zeros(self.mem_size, self.key_dim))
        self.register_buffer("values", torch.zeros(self.mem_size, self.pred_len, 1))
        self.register_buffer("valid", torch.zeros(self.mem_size, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))  # write pointer

    @torch.no_grad()
    def add(self, key: torch.Tensor, value: torch.Tensor):
        """
        key:   [B, key_dim] (normalized)
        value: [B, pred_len, 1]
        """
        B = key.size(0)
        for i in range(B):
            p = int(self.ptr.item())
            self.keys[p].copy_(key[i])
            self.values[p].copy_(value[i])
            self.valid[p] = True
            self.ptr[0] = (p + 1) % self.mem_size

    def retrieve(self, query_key: torch.Tensor) -> torch.Tensor:
        """
        query_key: [B, key_dim]
        return: y_mem [B, pred_len, 1]
        """
        if self.valid.sum() == 0:
            return torch.zeros(query_key.size(0), self.pred_len, 1, device=query_key.device, dtype=query_key.dtype)

        keys = self.keys[self.valid]     # [M, key_dim]
        vals = self.values[self.valid]   # [M, pred_len, 1]

        # cosine similarity because keys already normalized
        sim = query_key @ keys.t()       # [B, M]

        k = min(self.topk, sim.size(-1))
        top = torch.topk(sim, k=k, dim=-1)
        idx = top.indices                # [B, k]
        w = torch.softmax(top.values, dim=-1)  # [B, k]

        # gather values: [B, k, pred_len, 1] -> weighted sum -> [B, pred_len, 1]
        v = vals[idx]                    # advanced indexing
        y_mem = (v * w.view(-1, k, 1, 1)).sum(dim=1)
        return y_mem


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
        self.num_experts = 3

        self.teacher_forcing = True

        # -------- Embedding + pred tokens --------
        dropout = 0.3
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        # -------- backbone --------
        d_ff = 256
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
        self.router = SampleRouterFromX(c_in=4, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- heads --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=dropout),
            ExtremeHead(d_model, hidden=2*d_model, dropout=dropout),
        ])
        
        
        # -------- top-k gating --------
        self.top_k = int(getattr(config, "top_k", 2))          # top2
        self.router_tau = float(getattr(config, "router_tau", 1.0))  # softmax 温度
        self.tf_blend = float(getattr(config, "tf_blend", 0.0))      # 0=不使用GMM强制混合(避免“硬编码”)

        # -------- GMM label slices--------
        self.gmm_pt_start  = 2
        self.gmm_pt_end    = 5
        self.gmm_seq_start = 7
        self.gmm_seq_end   = 10
        
        # =========================================================
        # Turning-Point Memory (specialized for normalized 1st-diff in channel 0)
        # =========================================================
        self.use_memory = use_memory
        self.tp_target_idx = int(getattr(config, "tp_target_idx", 0))   # 0维是差分
        self.tp_key_len = int(getattr(config, "tp_key_len", min(32, self.seq_len)))
        self.tp_topk = int(getattr(config, "tp_topk", 8))
        self.tp_beta = float(getattr(config, "tp_beta", 0.3))           # memory residual 强度（建议 0.1~0.5）

        # turning detection thresholds (in normalized units)
        self.tp_eps = float(getattr(config, "tp_eps", 0.05))            # 死区，抑制0附近抖动
        self.tp_min_abs = float(getattr(config, "tp_min_abs", 0.6))     # |Δ|幅度阈值
        self.tp_min_jump = float(getattr(config, "tp_min_jump", 0.6))   # |Δ_t-Δ_{t-1}|阈值
        self.tp_future_region = int(getattr(config, "tp_future_region", self.pred_len))  # 只关心未来pred_len内的拐点

        # gating: only apply memory when "turning score" is high
        self.tp_score_thr = float(getattr(config, "tp_score_thr", 1.2))
        self.tp_score_temp = float(getattr(config, "tp_score_temp", 0.5))

        if self.use_memory:
            key_dim = int(getattr(config, "tp_key_dim", 64))
            key_hidden = int(getattr(config, "tp_key_hidden", 128))
            key_drop = float(getattr(config, "tp_key_dropout", 0.0))
            mem_size = int(getattr(config, "tp_mem_size", 4096))

            # 这里只用 target channel 做拐点形状；如果要多通道，把 in_ch 改成 len(tp_key_channels)
            self.tp_key_encoder = TurningPointKeyEncoder(in_ch=1, key_dim=key_dim, hidden=key_hidden, dropout=key_drop)
            self.tp_memory = TurningPointMemoryBank(mem_size=mem_size, key_dim=key_dim, pred_len=self.pred_len, topk=self.tp_topk)


    def _get_gmm_weight(self, x: torch.Tensor):
        pt  = x[:, :, self.gmm_pt_start:self.gmm_pt_end]
        seq = x[:, :, self.gmm_seq_start:self.gmm_seq_end]
        gmm = seq + 0.4 * pt
        gmm = gmm.abs().max(dim=1).values
        return gmm
    

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
    

    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        
        B = x.size(0)
        
        x1 = x[:,:,0:2]
        x2 = x[:,:,3:5]
        router_x = torch.cat([x1, x2], dim=-1)   # [B, L, 2+3+3] = [B, L, 8]
        
        # ---------------- routing ----------------
        router_logits = self.router(router_x)  # [B, E]
        gmm_logits = self._get_gmm_weight(x)
        
        logits = router_logits + gmm_logits
        # router 概率（带温度）
        tau = max(self.router_tau, 1e-6)
        router_prob = torch.softmax(logits / tau, dim=-1)  # [B, E]

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
        
        # =========================================================
        # Turning-Point Memory: write (train) + retrieve (train/test) + residual fuse
        # =========================================================
        if self.use_memory:
            # -------- query key from last tp_key_len history window (target diff channel) --------
            x_tgt = x[:, :, self.tp_target_idx:self.tp_target_idx + 1]                    # [B, seq_len, 1]
            x_win = x_tgt[:, -self.tp_key_len:, :]                                        # [B, key_len, 1]
            q_key = self.tp_key_encoder(x_win)                                            # [B, key_dim]

            # -------- retrieve memory prediction (diff trajectory) --------
            y_mem = self.tp_memory.retrieve(q_key)                                       # [B, pred_len, 1]

            # -------- compute a "turningness" score from recent history (no future needed) --------
            # use last (tp_key_len) window to estimate turning likelihood
            has_hist_tp, hist_score = _turning_score_from_diff(
                x_win.squeeze(-1),
                eps=self.tp_eps,
                min_abs=self.tp_min_abs,
                min_jump=self.tp_min_jump,
                region_start=max(1, self.tp_key_len - 8),   # 只看窗口尾部更贴近“即将拐”
                region_end=self.tp_key_len - 1
            )

            # gating coefficient beta in [0, tp_beta]
            beta = self.tp_beta * torch.sigmoid((hist_score - self.tp_score_thr) / max(self.tp_score_temp, 1e-6))
            y = y + beta.view(B, 1, 1) * y_mem

            # -------- write memory during training when y_true is available --------
            # y_true expected to be future diff of target variable: [B, pred_len, 1] or [B, pred_len, D]
            if self.training and (y_true is not None):
                if y_true.dim() == 3 and y_true.size(-1) != 1:
                    y_tgt = y_true[:, :, 0:1]  # 默认取第0列作为预测目标；如不同请改成你的target列
                else:
                    y_tgt = y_true

                # detect turning point in the FUTURE horizon (more meaningful for building the library)
                d_all = torch.cat([x_tgt.squeeze(-1), y_tgt.squeeze(-1)], dim=1)  # [B, seq_len + pred_len]
                # region: focus on boundary -> within next pred_len steps
                region_start = self.seq_len
                region_end = self.seq_len + min(self.tp_future_region, self.pred_len) - 1
                has_fut_tp, fut_score = _turning_score_from_diff(
                    d_all,
                    eps=self.tp_eps,
                    min_abs=self.tp_min_abs,
                    min_jump=self.tp_min_jump,
                    region_start=region_start,
                    region_end=region_end
                )

                # only write strong turning cases
                write_mask = has_fut_tp & (fut_score > self.tp_score_thr)
                if write_mask.any():
                    self.tp_memory.add(q_key[write_mask].detach(), y_tgt[write_mask].detach())

        return y