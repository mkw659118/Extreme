#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTM (Residual-Extreme Expert + Safer Memory Write)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.embedding import DataEmbedding


# =========================================================
# Heads
# =========================================================
class NormalHead(nn.Module):
    """Baseline head: d_model -> 1"""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, pred_len, d_model]
        return self.proj(x)  # [B, pred_len, 1]


class MidHead(nn.Module):
    """Baseline head: (d_model -> hidden -> 1)"""
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.fc = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x):  # [B, pred_len, d_model]
        x = self.drop(self.act(self.fc(x)))  # [B, pred_len, hidden]
        return self.proj(x)                  # [B, pred_len, 1]


class ResidualExtremeHead(nn.Module):
    """
    Extreme expert outputs ONLY residual correction Δy (not a full forecast).
    This is usually more RMSE-friendly on diff-normalized series.
    """
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # [B, pred_len, d_model]
        return self.net(x)  # [B, pred_len, 1]  (Δy)


# =========================================================
# Turning-point score
# =========================================================
def _turning_score_from_diff(
    diff_1d: torch.Tensor,
    eps: float = 0.05,
    min_abs: float = 0.6,
    min_jump: float = 0.6,
    region_start: int = 0,
    region_end: int = None,
):
    """
    diff_1d: [B, T]
    returns:
      has_tp: [B] bool
      score : [B] float
    """
    B, T = diff_1d.shape
    if region_end is None:
        region_end = T - 1
    region_end = min(region_end, T - 1)

    s = torch.sign(diff_1d)
    s = torch.where(diff_1d.abs() < eps, torch.zeros_like(s), s)

    flip = (s[:, 1:] * s[:, :-1] < 0)  # [B, T-1]

    d0 = diff_1d[:, :-1]  # [B, T-1]
    d1 = diff_1d[:, 1:]   # [B, T-1]
    amp_ok = (d0.abs() > min_abs) & (d1.abs() > min_abs)
    jump_ok = ((d1 - d0).abs() > min_jump)
    valid = flip & amp_ok & jump_ok

    t = torch.arange(1, T, device=diff_1d.device)  # [T-1]
    region = (t >= region_start) & (t <= region_end)
    valid = valid & region.view(1, -1)

    score_each = (d0.abs() + d1.abs() + (d1 - d0).abs())  # [B, T-1]
    score_each = torch.where(valid, score_each, torch.zeros_like(score_each))

    score, _ = score_each.max(dim=1)  # [B]
    has_tp = score > 0
    return has_tp, score


# =========================================================
# Memory Key Encoder + Memory Bank
# =========================================================
class TurningPointKeyEncoder(nn.Module):
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
        return key: [B, key_dim] (L2-normalized)
        """
        mean = x_win.mean(dim=1)                           # [B, in_ch]
        std = x_win.std(dim=1, unbiased=False)            # [B, in_ch]
        max_abs = x_win.abs().amax(dim=1)                 # [B, in_ch]
        last = x_win[:, -1, :]                            # [B, in_ch]
        feat = torch.cat([mean, std, max_abs, last], dim=-1)  # [B, 4*in_ch]
        key = self.net(feat)                              # [B, key_dim]
        key = F.normalize(key, dim=-1)
        return key


class TurningPointMemoryBank(nn.Module):
    """
    Ring-buffer memory:
      keys:   [N, key_dim]
      values: [N, pred_len, 1]  (store residual trajectory)
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
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def add(self, key: torch.Tensor, residual: torch.Tensor):
        """
        key:      [B, key_dim]
        residual: [B, pred_len, 1]
        """
        B = key.size(0)
        for i in range(B):
            p = int(self.ptr.item())
            self.keys[p].copy_(key[i])
            self.values[p].copy_(residual[i])
            self.valid[p] = True
            self.ptr[0] = (p + 1) % self.mem_size

    def retrieve(self, query_key: torch.Tensor) -> torch.Tensor:
        """
        query_key: [B, key_dim]
        return:    [B, pred_len, 1]
        """
        if int(self.valid.sum().item()) == 0:
            return torch.zeros(
                query_key.size(0), self.pred_len, 1,
                device=query_key.device, dtype=query_key.dtype
            )

        keys = self.keys[self.valid]      # [M, key_dim]
        vals = self.values[self.valid]    # [M, pred_len, 1]

        sim = query_key @ keys.t()        # [B, M]
        k = min(self.topk, sim.size(-1))
        top = torch.topk(sim, k=k, dim=-1)
        idx = top.indices                 # [B, k]
        w = torch.softmax(top.values, dim=-1)  # [B, k]

        v = vals[idx]                     # [B, k, pred_len, 1]
        r_mem = (v * w.view(-1, k, 1, 1)).sum(dim=1)  # [B, pred_len, 1]
        return r_mem


# =========================================================
# Router + CrossAttention
# =========================================================
class SampleRouterFromX(nn.Module):
    """
    Router features: std + max_abs + last over time dim.
    Input x: [B, L, C]
    Output logits: [B, E]
    """
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        in_dim = 3 * c_in
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=1, unbiased=False)      # [B, C]
        max_abs = x.abs().amax(dim=1)           # [B, C]
        last = x[:, -1, :]                      # [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)                   # [B, E]


class CrossAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V):
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        return ctx, attn


# =========================================================
# ExtremeLSTM
# =========================================================
class ExtremeLSTM(nn.Module):
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
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.d_model = int(d_model)
        self.c_in = int(c_in)

        # safe defaults if config is None
        cfg_dropout = 0.0 if config is None else float(getattr(config, "dropout", 0.0))
        self.dropout = cfg_dropout

        # -------- expert definition --------
        self.num_experts = 3  # 0: normal (baseline), 1: mid (baseline), 2: extreme (residual)

        # -------- Embedding + pred tokens --------
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        enc_layers = int(num_layers_intra_patch)
        dec_layers = int(num_layers_inter_patch)

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=enc_layers,
            batch_first=True,
            dropout=self.dropout if enc_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=self.dropout if dec_layers > 1 else 0.0,
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)

        # -------- router --------
        self.router = SampleRouterFromX(
            c_in=c_in,
            num_experts=self.num_experts,
            hidden=self.d_model,
            dropout=self.dropout,
        )

        # -------- heads (Residual Extreme Expert) --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),                                          # y0
            MidHead(d_model, hidden=d_model, dropout=self.dropout),        # y1
            ResidualExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),  # Δy
        ])

        # =========================================================
        # Turning-Point Memory
        # =========================================================
        self.use_memory = bool(use_memory)

        # read tp params safely even if config None
        def _cfg(name, default):
            return default if self.config is None else getattr(self.config, name, default)

        self.tp_target_idx = int(_cfg("tp_target_idx", 0))
        self.tp_key_len = int(_cfg("tp_key_len", min(32, self.seq_len)))
        self.tp_topk = int(_cfg("tp_topk", 8))
        self.tp_beta = float(_cfg("tp_beta", 0.3))

        self.tp_eps = float(_cfg("tp_eps", 0.05))
        self.tp_min_abs = float(_cfg("tp_min_abs", 0.6))
        self.tp_min_jump = float(_cfg("tp_min_jump", 0.6))

        self.tp_score_thr = float(_cfg("tp_score_thr", 1.2))
        self.tp_score_temp = float(_cfg("tp_score_temp", 0.5))

        if self.use_memory:
            key_dim = int(_cfg("tp_key_dim", 64))
            key_hidden = int(_cfg("tp_key_hidden", 128))
            key_drop = float(_cfg("tp_key_dropout", 0.0))
            mem_size = int(_cfg("tp_mem_size", 1024))

            self.tp_key_encoder = TurningPointKeyEncoder(
                in_ch=1, key_dim=key_dim, hidden=key_hidden, dropout=key_drop
            )
            self.tp_memory = TurningPointMemoryBank(
                mem_size=mem_size, key_dim=key_dim, pred_len=self.pred_len, topk=self.tp_topk
            )

    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]  (diff + normalized)
        return y: [B, pred_len, 1]  (diff + normalized)
        """
        B = x.size(0)

        # ---------------- routing ----------------
        router_logits = self.router(x)                       # [B, 3]
        router_prob = torch.softmax(router_logits, dim=-1)    # [B, 3]

        # ---------------- embedding ----------------
        x_emb_hist = self.embedding(x)                       # [B, seq_len, d_model]

        # ---------------- LSTM backbone ----------------
        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)       # enc_out: [B, seq_len, d_model]

        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        dec_out, _ = self.decoder(pred_token, (h_n, c_n))             # [B, pred_len, d_model]

        ctx, _ = self.xattn(dec_out, enc_out, enc_out)        # [B, pred_len, d_model]
        fused = torch.cat([dec_out, ctx], dim=-1)             # [B, pred_len, 2*d_model]
        fused = self.fuse_proj(fused)                         # [B, pred_len, d_model]
        final_shared = self.post_norm(fused)                  # [B, pred_len, d_model]

        # =========================================================
        # Residual MoE fusion:
        #   y0, y1 are baselines; delta is residual correction
        #   y = y_base + w_ext * delta
        # =========================================================
        y0 = self.expert_heads[0](final_shared)               # [B, pred_len, 1]
        y1 = self.expert_heads[1](final_shared)               # [B, pred_len, 1]
        delta = self.expert_heads[2](final_shared)            # [B, pred_len, 1]

        w0 = router_prob[:, 0].view(B, 1, 1)                  # [B,1,1]
        w1 = router_prob[:, 1].view(B, 1, 1)                  # [B,1,1]
        w_ext = router_prob[:, 2].view(B, 1, 1)               # [B,1,1]

        # normalize baseline weights so baselines aren't "diluted" by w_ext
        w_base_sum = (w0 + w1).clamp_min(1e-8)
        w0n = w0 / w_base_sum
        w1n = w1 / w_base_sum

        y_base = w0n * y0 + w1n * y1                          # [B, pred_len, 1]
        y = y_base + w_ext * delta                            # [B, pred_len, 1]

        # =========================================================
        # Turning-Point Memory (safer write: residual vs pre-mem y)
        # =========================================================
        if self.use_memory:
            y_pre_mem = y  # keep a copy BEFORE memory correction

            # target channel for TP key
            x_tgt = x[:, :, self.tp_target_idx:self.tp_target_idx + 1]  # [B, seq_len, 1]
            x_win = x_tgt[:, -self.tp_key_len:, :]                      # [B, key_len, 1]
            q_key = self.tp_key_encoder(x_win)                          # [B, key_dim]
            r_mem = self.tp_memory.retrieve(q_key)                      # [B, pred_len, 1]

            has_hist_tp, hist_score = _turning_score_from_diff(
                x_win.squeeze(-1),  # [B, key_len]
                eps=self.tp_eps,
                min_abs=self.tp_min_abs,
                min_jump=self.tp_min_jump,
                region_start=max(1, self.tp_key_len - 8),
                region_end=self.tp_key_len - 1
            )

            beta = self.tp_beta * torch.sigmoid(
                (hist_score - self.tp_score_thr) / max(self.tp_score_temp, 1e-6)
            )  # [B]

            # hard mask: no turning point => no memory perturbation (often lowers overall RMSE)
            mask = has_hist_tp.float().view(B, 1, 1)                    # [B,1,1]
            y = y_pre_mem + (beta.view(B, 1, 1) * mask) * r_mem         # [B, pred_len, 1]

            # training write: store residual vs y_pre_mem (NOT vs y after memory)
            if self.training and (y_true is not None):
                y_tgt = y_true[:, :, 0:1]                               # [B, pred_len, 1]
                res = (y_tgt - y_pre_mem.detach())                      # [B, pred_len, 1]
                self.tp_memory.add(q_key.detach(), res.detach())

        return y
