#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer
#           Shared Backbone + Top1 Expert Heads (sample-level routing)
#           + Safer Extreme/Turning-Point MemoryBank (for test-time robustness)

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


# =========================================================
# Module 0) Utility: Causal window mask
# =========================================================
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


# =========================================================
# Module 1) Standard Transformer Block 
# =========================================================
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
# Module 2) Router (sample-level) from RAW x
# =========================================================
class SampleRouterFromX(nn.Module):
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x):
        """
        x: [B, seq_len, c_in]
        returns logits: [B, E]
        """
        feat = x.mean(dim=1)  # [B, c_in]
        return self.net(feat)


# =========================================================
# Module 3) Expert Heads (E heads)
# =========================================================
class ExpertHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, tail_tokens: torch.Tensor):
        return self.proj(tail_tokens)  # [B, pred_len, 1]


# =========================================================
# Module 4) Diversity regularizer for heads
# =========================================================
def head_diversity_loss(expert_heads: nn.ModuleList, eps: float = 1e-8):
    """
    For Linear head: weight shape [1, d_model]
    Penalize pairwise cosine similarity (squared).
    """
    W = []
    for h in expert_heads:
        w = h.proj.weight.view(-1)  # [d_model]
        w = w / (w.norm(p=2) + eps)
        W.append(w)
    W = torch.stack(W, dim=0)  # [E, d_model]
    C = W @ W.t()              # [E, E]
    E = C.size(0)
    off = C - torch.eye(E, device=C.device, dtype=C.dtype)
    return (off ** 2).mean()


# =========================================================
# Module 5) Memory: safer key encoder (channel selection)
# =========================================================
class KeyEncoderSelected(nn.Module):
    """
    Encode x (selected channels) into a compact key.
    Pool: [mean, std, max_abs] over time -> MLP -> key

    Important:
      - by default EXCLUDES raw level channel (dim=5) to avoid "level matching"
      - focuses on shape features (delta/2nd-diff) and responsibilities (GMM)
    """
    def __init__(
        self,
        c_in: int,
        key_dim: int = 64,
        hidden: int = 128,
        dropout: float = 0.0,
        key_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.c_in = int(c_in)
        if key_indices is None or len(key_indices) == 0:
            key_indices = list(range(c_in))
        # keep unique and valid
        key_indices = sorted({int(i) for i in key_indices if 0 <= int(i) < c_in})
        self.key_indices = key_indices

        in_dim = 3 * len(self.key_indices)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, key_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        xs = x[:, :, self.key_indices]  # [B, L, C_sel]
        mean = xs.mean(dim=1)
        std = xs.std(dim=1, unbiased=False)
        max_abs = xs.abs().max(dim=1).values
        feat = torch.cat([mean, std, max_abs], dim=-1)
        return self.net(feat)


# =========================================================
# Module 6) Extreme/Turning-Point MemoryBank (robust version)
# =========================================================
class DiffExtremeMemoryBank(nn.Module):
    """
    Memory bank for extreme/turning-point samples.

    Store rule:
      score = max_abs(delta) + gamma * max_abs(second_diff(delta))
      threshold uses running EMA stats over training stream (not per-batch mean/std)

    Retrieve:
      cosine similarity topK
      confidence from max_sim vs sim_threshold
      correction default: RESIDUAL (safer for OOD test)

    Stores ONLY target_idx dim:
      y_true/y_pred can be [B,pred_len,D] -> stored as [B,pred_len,1] by selecting target_idx.
    """
    def __init__(
        self,
        c_in: int,
        pred_len: int,
        capacity: int = 1024,
        key_dim: int = 64,
        key_hidden: int = 128,
        key_dropout: float = 0.1,
        key_indices: Optional[List[int]] = None,
        top_k: int = 5,
        sim_threshold: float = 0.92,
        hard_threshold: float = 0.97,     # only used if mode="replace"
        beta: float = 0.3,
        mode: str = "residual",           # "residual" | "mix_true" | "replace"
        apply_during_train: bool = False, # usually False
        delta_idx: int = 0,
        second_diff_idx: Optional[int] = 9,  # you said dim=9 is second diff; use it directly if available
        score_k: float = 1.5,             # threshold = mu + score_k * std
        score_gamma: float = 0.5,
        min_store_ratio: float = 0.05,
        ema_momentum: float = 0.02,       # running stats update speed
        target_idx: int = 0,              # store first dim only
        match_expert: bool = False,
    ):
        super().__init__()
        assert mode in {"residual", "mix_true", "replace"}
        self.capacity = int(capacity)
        self.pred_len = int(pred_len)

        self.top_k = int(top_k)
        self.sim_threshold = float(sim_threshold)
        self.hard_threshold = float(hard_threshold)
        self.beta = float(beta)
        self.mode = mode
        self.apply_during_train = bool(apply_during_train)

        self.delta_idx = int(delta_idx)
        self.second_diff_idx = None if second_diff_idx is None else int(second_diff_idx)

        self.score_k = float(score_k)
        self.score_gamma = float(score_gamma)
        self.min_store_ratio = float(min_store_ratio)
        self.ema_momentum = float(ema_momentum)

        self.target_idx = int(target_idx)
        self.match_expert = bool(match_expert)

        self.encoder = KeyEncoderSelected(
            c_in=c_in, key_dim=key_dim, hidden=key_hidden, dropout=key_dropout, key_indices=key_indices
        )

        # buffers
        self.register_buffer("keys", torch.zeros(self.capacity, key_dim))
        self.register_buffer("vals_true", torch.zeros(self.capacity, self.pred_len, 1))
        self.register_buffer("vals_resid", torch.zeros(self.capacity, self.pred_len, 1))
        self.register_buffer("expert_ids", torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool))

        # running stats for score (EMA)
        self.register_buffer("score_mu", torch.tensor(0.0))
        self.register_buffer("score_var", torch.tensor(1.0))
        self.register_buffer("score_inited", torch.tensor(False))

        self.ptr = 0
        self.step = 0

    @torch.no_grad()
    def _take_target(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        if y.dim() == 1:
            y = y.view(1, -1, 1)
        elif y.dim() == 2:
            y = y.unsqueeze(-1)
        elif y.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported y dim={y.dim()}, shape={tuple(y.shape)}")

        # [B, pred_len, D]
        if y.size(1) != self.pred_len:
            L = y.size(1)
            if L > self.pred_len:
                y = y[:, :self.pred_len, :]
            else:
                pad = torch.zeros(y.size(0), self.pred_len - L, y.size(2), device=y.device, dtype=y.dtype)
                y = torch.cat([y, pad], dim=1)

        if y.size(-1) == 1:
            return y
        idx = max(0, min(self.target_idx, y.size(-1) - 1))
        return y[..., idx:idx + 1]

    @torch.no_grad()
    def _compute_score(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        delta = x[:, :, self.delta_idx]                     # [B,L]
        s1 = delta.abs().max(dim=1).values                  # [B]

        if self.second_diff_idx is not None and self.second_diff_idx < x.size(-1):
            d2 = x[:, :, self.second_diff_idx]
            s2 = d2.abs().max(dim=1).values
        else:
            # fallback: compute from delta
            if delta.size(1) >= 3:
                d2 = delta[:, 2:] - 2.0 * delta[:, 1:-1] + delta[:, :-2]
                s2 = d2.abs().max(dim=1).values
            else:
                s2 = torch.zeros_like(s1)

        return s1 + self.score_gamma * s2

    @torch.no_grad()
    def _update_running_stats(self, score: torch.Tensor):
        # score: [B]
        mu_b = score.mean()
        var_b = score.var(unbiased=False) + 1e-6

        if not bool(self.score_inited.item()):
            self.score_mu.copy_(mu_b)
            self.score_var.copy_(var_b)
            self.score_inited.copy_(torch.tensor(True, device=self.score_inited.device))
            return

        m = self.ema_momentum
        self.score_mu.copy_((1 - m) * self.score_mu + m * mu_b)
        self.score_var.copy_((1 - m) * self.score_var + m * var_b)

    @torch.no_grad()
    def _select_to_store(self, x: torch.Tensor) -> torch.Tensor:
        score = self._compute_score(x)          # [B]
        self._update_running_stats(score)

        std = torch.sqrt(self.score_var).clamp_min(1e-6)
        thr = self.score_mu + self.score_k * std

        mask = score >= thr

        # ensure at least min_store_ratio
        B = score.numel()
        min_k = max(1, int(B * self.min_store_ratio))
        if mask.sum().item() < min_k:
            top_idx = torch.topk(score, k=min_k, largest=True).indices
            new_mask = torch.zeros_like(mask, dtype=torch.bool)
            new_mask[top_idx] = True
            mask = new_mask
        return mask

    @torch.no_grad()
    def update(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        expert_idx: Optional[torch.Tensor] = None
    ):
        self.step += 1

        y_pred = self._take_target(y_pred)
        y_true = self._take_target(y_true)
        if y_pred is None or y_true is None:
            return

        store_mask = self._select_to_store(x)
        if store_mask.sum().item() == 0:
            return

        x_sel = x[store_mask]
        y_pred_sel = y_pred[store_mask]             # [M,pred,1]
        y_true_sel = y_true[store_mask]             # [M,pred,1]
        resid_sel = (y_true_sel - y_pred_sel)       # [M,pred,1]

        keys = self.encoder(x_sel)
        keys = F.normalize(keys, dim=-1)

        if expert_idx is None:
            expert_sel = torch.full((keys.size(0),), -1, device=keys.device, dtype=torch.long)
        else:
            expert_sel = expert_idx[store_mask].detach().to(keys.device).long()

        for i in range(keys.size(0)):
            j = self.ptr
            self.keys[j].copy_(keys[i])
            self.vals_true[j].copy_(y_true_sel[i])
            self.vals_resid[j].copy_(resid_sel[i])
            self.expert_ids[j] = expert_sel[i]
            self.valid[j] = True
            self.ptr = (self.ptr + 1) % self.capacity

    @torch.no_grad()
    def retrieve(self, x: torch.Tensor, expert_idx: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        B = x.size(0)
        if self.valid.sum().item() == 0:
            return None, None, torch.zeros(B, 1, 1, device=x.device)

        q = self.encoder(x)
        q = F.normalize(q, dim=-1)

        K = self.keys[self.valid]           # [N, key_dim]
        Vt = self.vals_true[self.valid]     # [N, pred, 1]
        Vr = self.vals_resid[self.valid]    # [N, pred, 1]
        eid = self.expert_ids[self.valid]   # [N]

        # Optional: filter by expert
        if self.match_expert and expert_idx is not None:
            mem_true = torch.zeros(B, self.pred_len, 1, device=x.device)
            mem_resid = torch.zeros(B, self.pred_len, 1, device=x.device)
            conf = torch.zeros(B, 1, 1, device=x.device)

            for b in range(B):
                mask = (eid == int(expert_idx[b].item()))
                if mask.sum().item() == 0:
                    mask = torch.ones_like(eid, dtype=torch.bool)

                Kb = K[mask]
                Vtb = Vt[mask]
                Vrb = Vr[mask]

                sim = (q[b:b+1] @ Kb.t()).squeeze(0)
                topk = min(self.top_k, sim.numel())
                s, idx = torch.topk(sim, k=topk, largest=True)
                w = torch.softmax(s, dim=0)

                mem_true[b] = (Vtb[idx] * w.view(-1, 1, 1)).sum(dim=0)
                mem_resid[b] = (Vrb[idx] * w.view(-1, 1, 1)).sum(dim=0)

                smax = s.max()
                c = ((smax - self.sim_threshold) / max(1e-6, (1.0 - self.sim_threshold))).clamp(0.0, 1.0)
                conf[b, 0, 0] = c
            return mem_true, mem_resid, conf

        # Vectorized retrieval
        sim = q @ K.t()  # [B,N]
        topk = min(self.top_k, sim.size(1))
        s, idx = torch.topk(sim, k=topk, dim=-1, largest=True)   # [B,topk]
        w = torch.softmax(s, dim=-1)                             # [B,topk]

        Vt_sel = Vt[idx]  # [B,topk,pred,1]
        Vr_sel = Vr[idx]
        mem_true = (Vt_sel * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        mem_resid = (Vr_sel * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        smax = s.max(dim=-1).values
        conf = ((smax - self.sim_threshold) / max(1e-6, (1.0 - self.sim_threshold))).clamp(0.0, 1.0)
        conf = conf.view(B, 1, 1)
        return mem_true, mem_resid, conf

    @torch.no_grad()
    def apply(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        expert_idx: Optional[torch.Tensor] = None,
        training: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_pred = self._take_target(y_pred)
        if y_pred is None:
            raise ValueError("y_pred is None")

        if training and (not self.apply_during_train):
            return y_pred, {
                "mem_conf_mean": torch.tensor(0.0, device=y_pred.device),
                "mem_alpha_mean": torch.tensor(0.0, device=y_pred.device),
            }

        mem_true, mem_resid, conf = self.retrieve(x, expert_idx=expert_idx)
        if mem_true is None or mem_resid is None:
            return y_pred, {
                "mem_conf_mean": torch.tensor(0.0, device=y_pred.device),
                "mem_alpha_mean": torch.tensor(0.0, device=y_pred.device),
            }

        alpha = (self.beta * conf).clamp(0.0, 1.0)  # [B,1,1]

        if self.mode == "replace":
            hard = (conf >= self.hard_threshold).float()
            y_out = (1.0 - hard) * y_pred + hard * mem_true
        elif self.mode == "mix_true":
            y_out = (1.0 - alpha) * y_pred + alpha * mem_true
        else:  # "residual" (recommended)
            y_out = y_pred + alpha * mem_resid

        return y_out, {
            "mem_conf_mean": conf.mean().detach(),
            "mem_alpha_mean": alpha.mean().detach(),
        }


# =========================================================
# Main Model
# =========================================================
class ThreeExpertPatchTransformer(nn.Module):
    """
    Shared backbone; experts only in prediction heads; router selects top-1 expert per sample.

    dim0: delta
    dim5: raw sequence
    dim9: second diff

    Memory in this version:
      - key EXCLUDES dim5 by default
      - uses residual correction by default 
      - stores only y[:, :, 0:1]
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

        # -------- losses weights --------
        self.w_router_ce = float(getattr(config, "w_router_ce", 1.0))
        self.w_balance   = float(getattr(config, "w_balance", 0.01))
        self.w_head_div  = float(getattr(config, "w_head_div", 0.01))

        self.teacher_forcing = bool(getattr(config, "teacher_forcing", True))

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



        # -------- GMM label slices (kept compatible with your original) --------
        self.gmm_pt_start  = int(getattr(config, "gmm_pt_start", 2))
        self.gmm_pt_end    = int(getattr(config, "gmm_pt_end", 5))
        self.gmm_seq_start = int(getattr(config, "gmm_seq_start", 6))
        self.gmm_seq_end   = int(getattr(config, "gmm_seq_end", 9))

        # -------- memory --------
        self.use_memory = bool(use_memory)
        self.memory: Optional[DiffExtremeMemoryBank] = None
        if self.use_memory:
            # Default key indices (exclude raw dim=5)
            # focus: delta(0), prob(1), gmm_pt(2-4), gmm_seq(6-8), second_diff(9)
            default_key_idx = []
            for idx in [0, 1, 9]:
                if 0 <= idx < c_in:
                    default_key_idx.append(idx)

            # gmm ranges if valid
            for idx in range(self.gmm_pt_start, self.gmm_pt_end):
                if 0 <= idx < c_in and idx != 5:
                    default_key_idx.append(idx)
            for idx in range(self.gmm_seq_start, self.gmm_seq_end):
                if 0 <= idx < c_in and idx != 5:
                    default_key_idx.append(idx)

            # allow override from config (list/tuple)
            cfg_key_idx = getattr(config, "mem_key_indices", None)
            if isinstance(cfg_key_idx, (list, tuple)) and len(cfg_key_idx) > 0:
                key_indices = list(cfg_key_idx)
            else:
                key_indices = default_key_idx

            self.memory = DiffExtremeMemoryBank(
                c_in=c_in,
                pred_len=pred_len,
                capacity=int(getattr(config, "mem_capacity", 1024)),
                key_dim=int(getattr(config, "mem_key_dim", 64)),
                key_hidden=int(getattr(config, "mem_key_hidden", 128)),
                key_dropout=float(getattr(config, "mem_key_dropout", 0.1)),
                key_indices=key_indices,
                top_k=int(getattr(config, "mem_top_k", 5)),
                sim_threshold=float(getattr(config, "mem_sim_threshold", 0.92)),
                hard_threshold=float(getattr(config, "mem_hard_threshold", 0.97)),
                beta=float(getattr(config, "mem_beta", 0.3)),
                mode="residual",     # default safer
                apply_during_train=bool(getattr(config, "mem_apply_train", False)),
                delta_idx=int(getattr(config, "mem_delta_idx", 0)),
                second_diff_idx=int(getattr(config, "mem_second_diff_idx", 9)),
                score_k=float(getattr(config, "mem_score_k", 1.5)),
                score_gamma=float(getattr(config, "mem_score_gamma", 0.5)),
                min_store_ratio=float(getattr(config, "mem_min_store_ratio", 0.05)),
                ema_momentum=float(getattr(config, "mem_ema_momentum", 0.02)),
                target_idx=int(getattr(config, "mem_target_idx", 0)),  # store first y-dim only
                match_expert=bool(getattr(config, "mem_match_expert", False)),
            )

        self.aux_loss_dict: Dict[str, torch.Tensor] = {}
        self.mem_stat_dict: Dict[str, torch.Tensor] = {}

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


    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        y_true (optional): [B, pred_len, D] or [B, pred_len, 1]
        """
        B = x.size(0)

        # ---------------- routing ----------------
        router_logits = self.router(x)                  # [B, E]
        router_prob = torch.softmax(router_logits, -1)  # [B, E]

        if route_labels is None:
            route_labels = self._gmm_argmax_label(x)  # [B]
        
        if self.training and self.teacher_forcing:
            expert_idx = route_labels
        else:
            expert_idx = torch.argmax(router_logits, dim=-1)

        # ---------------- embedding + pred tokens ----------------
        x_emb_hist = self.embedding(x)                               # [B, seq_len, d_model]
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)      # [B, pred_len, d_model]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)                # [B, total_len, d_model]

        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # ---------------- shared backbone ----------------
        final_shared = self.forward_backbone(x_emb, intra_mask, inter_mask)   # [B, total_len, d_model]
        final_shared = final_shared[:, -self.pred_len:, :]                             # [B, pred_len, d_model]

        # ---------------- heads ----------------
        y_all = torch.cat([h(final_shared) for h in self.expert_heads], dim=-1)         # [B, pred_len, E]
        idx = expert_idx.view(B, 1, 1).expand(B, self.pred_len, 1)
        y_base = y_all.gather(dim=-1, index=idx)                                # [B, pred_len, 1]

        # ---------------- aux losses ----------------
        aux_loss = {}
        if self.w_router_ce > 0.0 and self.training:
            aux_loss["router_ce"] = self.w_router_ce * F.cross_entropy(router_logits, route_labels)

        if self.w_balance > 0.0 and self.training:
            mean_p = router_prob.mean(dim=0)
            uniform = torch.full_like(mean_p, 1.0 / self.num_experts)
            aux_loss["balance"] = self.w_balance * ((mean_p - uniform) ** 2).sum()

        if self.w_head_div > 0.0 and self.training and self.num_experts > 1:
            aux_loss["head_div"] = self.w_head_div * head_diversity_loss(self.expert_heads)

        # ---------------- memory apply + update ----------------
        y = y_base
        mem_stat = {}

        if self.use_memory and self.memory is not None:
            # apply correction (default: only in eval; controlled by mem_apply_train)
            y, mem_stat = self.memory.apply(x, y_base, expert_idx=expert_idx, training=self.training)

            # update memory only during training when y_true is available
            if self.training and (y_true is not None):
                self.memory.update(
                    x=x,
                    y_pred=y_base.detach(),
                    y_true=y_true.detach(),
                    expert_idx=expert_idx.detach(),
                )

        self.aux_loss_dict = aux_loss
        self.mem_stat_dict = mem_stat
        return y