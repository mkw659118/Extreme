# =========================================================
# PatchExtremeMemoryTransformer + External MemoryBank (plug-in)
# - MemoryBank is an "external module" that you attach to the backbone.
# - In forward(), right before "return y", we retrieve and fuse: y = y + beta * y_mem
# - Optional: auto-write (store) into memory when y_true is provided in training.
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict, Tuple, List

from layers.embedding import DataEmbedding


# =========================================================
# External MemoryBank (standalone plug-in)
# =========================================================
class ExternalMemoryBank(nn.Module):
    """
    A simple retrieval memory:
      - keys:   [N, seq_len, C]
      - values: [N, pred_len, 1] (store ONLY one target channel from y_true)
    Retrieval uses cosine similarity over a chosen target channel of x (default target_idx=0).

    APIs:
      construct_index(num)
      add_key_value(x_enc, y_true, index=None)
      retrieval(x_enc, index=None, training=False) -> (y_mem[B,pred_len,1], sims[B,1,k])
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_in: int,
        retrieval_num: int,
        retrieval_stride: int = 1,
        use_norm: bool = False,
        x_target_idx: int = 0,
        y_target_idx: int = 0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.c_in = int(c_in)

        self.retrieval_num = int(retrieval_num)
        self.retrieval_stride = int(retrieval_stride)
        self.use_norm = bool(use_norm)

        self.x_target_idx = int(x_target_idx)  # which x channel used for similarity
        self.y_target_idx = int(y_target_idx)  # which y channel stored/retrieved

        # buffers (allocated by construct_index)
        self.keys = None           # [N, L, C]
        self.values = None         # [N, P, 1]
        self.value_cache = None    # [N, P] for fast gather
        self.capacity = 0
        self.size = 0              # filled size

    def construct_index(self, num: int, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype

        self.capacity = int(num)
        self.size = 0
        self.keys = torch.zeros(self.capacity, self.seq_len, self.c_in, device=device, dtype=dtype)
        self.values = torch.zeros(self.capacity, self.pred_len, 1, device=device, dtype=dtype)
        self.value_cache = None

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y_true: torch.Tensor, index: Optional[torch.Tensor] = None):
        """
        x_enc:  [B, seq_len, C]
        y_true: [B, pred_len, D] or [B, pred_len, 1]
        index:
          - None -> append
          - Tensor [B] -> scatter to positions
        """
        if self.keys is None or self.values is None or self.capacity <= 0:
            return

        B = x_enc.size(0)

        # ---- pick y target channel -> [B, pred_len, 1]
        if y_true is None:
            return
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1)
        if y_true.size(-1) == 1:
            y_store = y_true
        else:
            y_store = y_true[..., self.y_target_idx:self.y_target_idx + 1]

        # ---- optional normalization (same spirit as reference)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()                       # [B,1,C]
            x0 = x_enc - means
            stdev = torch.sqrt(torch.var(x0, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B,1,C]
            x_enc_n = x0 / stdev

            mu_y = means[:, :, self.x_target_idx:self.x_target_idx + 1]       # [B,1,1]
            sd_y = stdev[:, :, self.x_target_idx:self.x_target_idx + 1]       # [B,1,1]
            y_store = (y_store - mu_y) / sd_y
        else:
            x_enc_n = x_enc

        # ---- write keys/values
        if index is None:
            # append
            start = self.size
            end = min(self.size + B, self.capacity)
            write_B = end - start
            if write_B <= 0:
                return
            self.keys[start:end] = x_enc_n[:write_B]
            self.values[start:end] = y_store[:write_B]
            self.size = end
        else:
            idx = index.to(self.keys.device).long().view(-1)
            assert idx.numel() == B, "index must be shape [B]"
            self.keys[idx] = x_enc_n
            self.values[idx] = y_store
            self.size = max(self.size, int(idx.max().item()) + 1)

        # invalidate cache
        self.value_cache = None
        torch.cuda.empty_cache()

    def retrieval(self, x_enc: torch.Tensor, index: Optional[torch.Tensor] = None, training: bool = False):
        """
        x_enc: [B, seq_len, C] (should be normalized consistently with memory if use_norm=True)
        index: [B] optional (for neighbor masking)
        return:
          y_mem: [B, pred_len, 1]
          sims:  [B, 1, k]
        """
        B = x_enc.size(0)
        k = self.retrieval_num

        if self.keys is None or self.values is None or self.size <= 0:
            y_mem = torch.zeros(B, self.pred_len, 1, device=x_enc.device, dtype=x_enc.dtype)
            sims = torch.zeros(B, 1, k, device=x_enc.device, dtype=x_enc.dtype)
            return y_mem, sims

        N = self.size

        # similarity only on one chosen channel (target_idx)
        q = x_enc[:, :, self.x_target_idx]              # [B, L]
        K = self.keys[:N, :, self.x_target_idx]         # [N, L]

        dis = self._cosine_similarity_2d(q, K)          # [B, N]

        # neighbor mask (same idea as your reference; only works if index is meaningful)
        if training and (index is not None):
            idx0 = index.to(dis.device).long().view(-1)  # [B]
            self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=dis.device).unsqueeze(0)  # [1,2L+1]
            invalid = idx0.unsqueeze(1) + self_range
            invalid = invalid // max(1, self.retrieval_stride)
            invalid.clamp_(0, N - 1)
            row = torch.arange(B, device=dis.device).unsqueeze(1).expand_as(invalid)
            dis[row, invalid] = -100.0

        # topk
        dis_topk, idx_topk = torch.topk(dis, k=min(k, N), dim=1)   # [B,k]
        # pad if N<k
        if dis_topk.size(1) < k:
            pad = k - dis_topk.size(1)
            dis_topk = torch.cat([dis_topk, dis_topk.new_full((B, pad), -100.0)], dim=1)
            idx_topk = torch.cat([idx_topk, idx_topk.new_zeros((B, pad))], dim=1)

        sims = dis_topk.unsqueeze(1)                                # [B,1,k]
        probs = torch.softmax(dis_topk, dim=1).unsqueeze(-1)        # [B,k,1]

        # values cache: [N, pred_len]
        if self.value_cache is None or self.value_cache.size(0) != N:
            self.value_cache = self.values[:N, :, 0].contiguous()

        gathered = self.value_cache[idx_topk]                       # [B,k,pred_len]
        y_mem = torch.sum(probs * gathered, dim=1, keepdim=False)   # [B,pred_len]
        y_mem = y_mem.unsqueeze(-1)                                 # [B,pred_len,1]
        return y_mem, sims

    @staticmethod
    def _cosine_similarity_2d(q: torch.Tensor, K: torch.Tensor):
        # q: [B,L], K: [N,L] -> [B,N]
        qn = F.normalize(q, p=2, dim=-1)
        Kn = F.normalize(K, p=2, dim=-1)
        return qn @ Kn.t()


class MemoryFusionGate(nn.Module):
    """
    Compute beta in y = y + beta * y_mem
    Inputs:
      y_pred: [B, pred_len, 1]
      sims:   [B, 1, k]
    Output:
      beta:   [B, 1, 1] (broadcastable)
    """
    def __init__(self, retrieval_num: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        in_dim = int(retrieval_num) + 3  # sims(k) + stats(3)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, y_pred: torch.Tensor, sims: torch.Tensor):
        # sims: [B,1,k] -> [B,k]
        s = sims.squeeze(1)
        # y stats over pred_len: mean/std/max_abs -> [B,3]
        mean = y_pred.mean(dim=1).squeeze(-1)
        std = y_pred.std(dim=1, unbiased=False).squeeze(-1)
        max_abs = y_pred.abs().max(dim=1).values.squeeze(-1)
        feat = torch.cat([s, mean.unsqueeze(1), std.unsqueeze(1), max_abs.unsqueeze(1)], dim=1)  # [B,k+3]
        beta = self.net(feat).view(-1, 1, 1)
        return beta


# =========================================================
# Your original model code (only minimal changes: attach & fuse memory)
# =========================================================
class NormalHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x)


class MidHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or d_model
        self.fc = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.drop(self.act(self.fc(x)))
        return self.proj(x)


class ExtremeHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = None, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or (2 * d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return self.proj(x)


def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    if win_size is None or win_size <= 0 or win_size > seq_len:
        win_size = max(1, seq_len // 2)

    upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)

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
        feat = x.mean(dim=1)
        return self.net(feat)


def head_diversity_loss(expert_heads: nn.ModuleList, eps: float = 1e-8):
    W = []
    for h in expert_heads:
        w = h.proj.weight.view(-1)
        w = w / (w.norm(p=2) + eps)
        W.append(w)
    W = torch.stack(W, dim=0)
    C = W @ W.t()
    E = C.size(0)
    off = C - torch.eye(E, device=C.device, dtype=C.dtype)
    return (off ** 2).mean()


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

        assert self.total_len % self.patch_len == 0
        self.num_patches = self.total_len // self.patch_len

        # -------- expert definition --------
        self.gmm_start = int(getattr(config, "gmm_start", 2))
        self.gmm_end   = int(getattr(config, "gmm_end", 5))
        assert 0 <= self.gmm_start < self.gmm_end <= c_in
        self.num_experts = int(self.gmm_end - self.gmm_start)
        assert self.num_experts == 3, "此处示例按3专家写的"

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

        # -------- heads --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=dropout),
            ExtremeHead(d_model, hidden=2*d_model, dropout=dropout),
        ])

        label2expert = getattr(config, "label2expert", None)
        if label2expert is None:
            label2expert = list(range(self.num_experts))
        self.register_buffer("label2expert", torch.tensor(label2expert, dtype=torch.long))

        # -------- GMM label slices --------
        self.gmm_pt_start  = int(getattr(config, "gmm_pt_start", 2))
        self.gmm_pt_end    = int(getattr(config, "gmm_pt_end", 5))
        self.gmm_seq_start = int(getattr(config, "gmm_seq_start", 6))
        self.gmm_seq_end   = int(getattr(config, "gmm_seq_end", 9))

        self.aux_loss_dict: Dict[str, torch.Tensor] = {}

        # =====================================================
        # Plug-in MemoryBank (EXTERNAL MODULE)
        # =====================================================
        self.use_memory = bool(use_memory) and bool(getattr(config, "use_memory", True))
        self.memory_write_in_forward = bool(getattr(config, "memory_write_in_forward", True))

        if self.use_memory:
            # which x channel defines similarity + which y channel stored
            x_target_idx = int(getattr(config, "mem_x_target_idx", 0))
            y_target_idx = int(getattr(config, "mem_y_target_idx", 0))

            retrieval_num = int(getattr(config, "retrieval_num", 8))
            retrieval_stride = int(getattr(config, "retrieval_stride", 1))
            mem_use_norm = bool(getattr(config, "mem_use_norm", False))  # independent switch

            self.memory_bank = ExternalMemoryBank(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                c_in=self.c_in,
                retrieval_num=retrieval_num,
                retrieval_stride=retrieval_stride,
                use_norm=mem_use_norm,
                x_target_idx=x_target_idx,
                y_target_idx=y_target_idx,
            )

            gate_hidden = int(getattr(config, "mem_gate_hidden", 64))
            self.memory_gate = MemoryFusionGate(retrieval_num=retrieval_num, hidden=gate_hidden, dropout=dropout)
        else:
            self.memory_bank = None
            self.memory_gate = None

    # -------- convenience wrappers (so you can treat memory as “external library”) --------
    def construct_index(self, num: int):
        if self.use_memory and self.memory_bank is not None:
            self.memory_bank.construct_index(num, device=self.pred_tokens.device, dtype=self.pred_tokens.dtype)

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y_true: torch.Tensor, index: Optional[torch.Tensor] = None):
        if self.use_memory and self.memory_bank is not None:
            self.memory_bank.add_key_value(x_enc, y_true, index=index)

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
        sample_ids: [B] (optional, used as "index" for neighbor mask in training)
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
        x_emb_hist = self.embedding(x)                                     # [B, seq_len, d_model]
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)       # [B, pred_len, d_model]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)                 # [B, total_len, d_model]

        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # ---------------- shared backbone ----------------
        final_shared = self.forward_backbone(x_emb, intra_mask, inter_mask)    # [B, total_len, d_model]
        final_shared = final_shared[:, -self.pred_len:, :]                      # [B, pred_len, d_model]

        # ---------------- heads ----------------
        y_all = torch.cat([h(final_shared) for h in self.expert_heads], dim=-1) # [B, pred_len, E]
        idx = expert_idx.view(B, 1, 1).expand(B, self.pred_len, 1)
        y = y_all.gather(dim=-1, index=idx)                                     # [B, pred_len, 1]

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

        self.aux_loss_dict = aux_loss

        # =====================================================
        # MemoryBank fusion (RIGHT BEFORE RETURN)
        #   y <- y + beta * y_mem
        # =====================================================
        if self.use_memory and (self.memory_bank is not None) and (self.memory_bank.size > 0):
            # IMPORTANT: retrieval uses x (raw) consistently; if you want normalized x, normalize before calling
            y_mem, sims = self.memory_bank.retrieval(
                x_enc=x,
                index=sample_ids,
                training=self.training
            )  # y_mem [B,pred_len,1], sims [B,1,k]
            beta = self.memory_gate(y, sims)  # [B,1,1]
            y = y + beta * y_mem

        # optional auto-write (store) AFTER retrieval to avoid immediate self-match
        if self.use_memory and self.memory_write_in_forward and self.training and (y_true is not None):
            self.add_key_value(x_enc=x, y_true=y_true, index=sample_ids)

        return y
