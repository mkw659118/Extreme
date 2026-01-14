
# #Author  :   mkw
# #Time    :   2025/09/17 10:50:52
# #Desc    :   PatchExtremeMemoryTransformer (Scheme C: MoE in FFN layers, NO Memory)
# #           - Attention is shared (standard Transformer attention)
# #           - FFN is replaced by MoE-FFN (Top-k experts) inside each Transformer block
# #           - GMM prior weights are PROVIDED by x (already in x channels)
# #           - Router logits are computed from token embeddings (inside MoE-FFN)
# #           - Fuse: concat([log(GMM_prior), router_logits]) -> FusionGate -> final gating logits
# #           - Top-k select FFN experts per token, then weighted sum as FFN output
# #           - Final prediction uses a SINGLE head on the tail tokens (not multi-head experts)

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from layers.embedding import DataEmbedding
# from modules.IFT_EncDec import ImplicitForecaster  # kept for compatibility; not used here


# # =========================================================
# # Module 0) Utility: Causal window mask
# # =========================================================
# def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
#     if win_size is None or win_size <= 0 or win_size > seq_len:
#         win_size = max(1, seq_len // 2)

#     upper = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)

#     # left-window clipping
#     if win_size < seq_len:
#         for i in range(seq_len):
#             left = max(0, i - win_size + 1)
#             upper[i, :left] = True

#     # additive mask: visible=0, masked=-inf
#     attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
#     attn_bias.masked_fill_(upper, float("-inf"))
#     return attn_bias


# # =========================================================
# # Module 1) Router Network (token-wise gating logits)
# #   Input : tokens [B, L, d_model]
# #   Output: router_logits [B, L, E]
# # =========================================================
# class RouterMLP(nn.Module):
#     def __init__(self, d_model: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, num_experts),
#         )

#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         return self.net(tokens)  # [B, L, E]


# # =========================================================
# # Module 2) FusionGate
# #   Fuse GMM prior (from x) and router logits to final gating logits.
# #   Input : log_q  [B, L, E]  (log of normalized GMM weights)
# #           r      [B, L, E]  (router logits)
# #   Output: s      [B, L, E]  (final gating logits)
# # =========================================================
# class FusionGate(nn.Module):
#     def __init__(self, num_experts: int, hidden: int = 64):
#         super().__init__()
#         E = num_experts
#         self.fuse = nn.Sequential(
#             nn.Linear(2 * E, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, E),
#         )

#     def forward(self, log_q: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
#         z = torch.cat([log_q, router_logits], dim=-1)  # [B, L, 2E]
#         return self.fuse(z)                             # [B, L, E]


# # =========================================================
# # Module 3) Top-k sparse softmax on expert dimension
# # =========================================================
# def topk_sparse_softmax(logits: torch.Tensor, k: int, tau: torch.Tensor):
#     """
#     logits: [B, L, E]
#     tau   : scalar tensor
#     return:
#       top_idx: [B, L, k]
#       top_w  : [B, L, k]  (softmax over top-k)
#     """
#     B, L, E = logits.shape
#     k = min(int(k), int(E))
#     topv, topi = torch.topk(logits, k=k, dim=-1)  # [B, L, k]
#     tau = tau.clamp_min(1e-6)
#     topw = torch.softmax(topv / tau, dim=-1)      # [B, L, k]
#     return topi, topw


# # =========================================================
# # Module 4) Expert FFN (one expert)
# # =========================================================
# class ExpertFFN(nn.Module):
#     def __init__(self, d_model: int, d_ff: int, dropout: float = 0.3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)  # [B, L, d_model]


# # =========================================================
# # Module 5) MoE-FFN (Scheme C core)
# #   - Experts: E separate FFNs
# #   - Router: token-wise logits
# #   - GMM prior: from x, provided as log_q
# #   - Fusion: concat([log_q, router_logits]) -> gate_logits
# #   - Top-k select experts -> weighted sum = FFN output
# # =========================================================
# class MoEFeedForward(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         d_ff: int,
#         num_experts: int,
#         topk: int,
#         router_hidden: int = 128,
#         router_dropout: float = 0.0,
#         gate_hidden: int = 64,
#         dropout: float = 0.3,
#         lb_lambda: float = 0.0,   # optional load-balance loss weight
#         eps: float = 1e-8
#     ):
#         super().__init__()
#         self.E = int(num_experts)
#         self.topk = int(topk)
#         self.eps = eps
#         self.lb_lambda = float(lb_lambda)

#         # Experts
#         self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=dropout) for _ in range(self.E)])

#         # Router + Fusion gate
#         self.router = RouterMLP(d_model=d_model, num_experts=self.E, hidden=router_hidden, dropout=router_dropout)
#         self.fusion_gate = FusionGate(num_experts=self.E, hidden=gate_hidden)

#         # Temperature for sparse softmax (trainable)
#         self.tau_raw = nn.Parameter(torch.tensor(0.0))  # tau = softplus + 1e-3

#         # Expose aux loss (optional)
#         self.aux_loss = torch.tensor(0.0)

#     def forward(self, x: torch.Tensor, log_q: torch.Tensor):
#         """
#         x:     [B, L, d_model]
#         log_q: [B, L, E] (log of GMM prior probs)
#         return:
#           y: [B, L, d_model] (MoE-FFN output)
#         """
#         B, L, D = x.shape
#         assert log_q.shape[:2] == (B, L) and log_q.shape[2] == self.E, "log_q shape mismatch"

#         # (1) router logits from token embeddings
#         router_logits = self.router(x)  # [B, L, E]

#         # (2) fusion -> final gating logits
#         gate_logits = self.fusion_gate(log_q, router_logits)  # [B, L, E]

#         # (3) Top-k sparse gating
#         tau = F.softplus(self.tau_raw) + 1e-3
#         topi, topw = topk_sparse_softmax(gate_logits, k=self.topk, tau=tau)  # [B,L,k], [B,L,k]

#         # (4) compute all expert FFN outputs (E is small in your case; OK)
#         # expert_outs: [B, L, E, D]
#         expert_outs = torch.stack([exp(x) for exp in self.experts], dim=2)

#         # (5) gather selected experts: [B,L,k,D]
#         idx = topi.unsqueeze(-1).expand(-1, -1, -1, D)              # [B,L,k,D]
#         sel = expert_outs.gather(dim=2, index=idx)                  # [B,L,k,D]

#         # (6) weighted sum over top-k -> [B,L,D]
#         y = (topw.unsqueeze(-1) * sel).sum(dim=2)

#         # (Optional) load-balance loss to reduce expert collapse
#         # Use full softmax distribution (before top-k) as "importance"
#         if self.lb_lambda > 0.0 and self.training:
#             p = torch.softmax(gate_logits, dim=-1)                  # [B,L,E]
#             mean_p = p.mean(dim=(0, 1))                             # [E]
#             uniform = torch.full_like(mean_p, 1.0 / self.E)
#             self.aux_loss = self.lb_lambda * ((mean_p - uniform) ** 2).sum()
#         else:
#             self.aux_loss = torch.tensor(0.0, device=x.device)

#         return y


# # =========================================================
# # Module 6) Transformer block with MoE-FFN
# #   - Attention shared
# #   - FFN replaced by MoEFeedForward
# # =========================================================
# class TransformerMoEBlock(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         d_ff: int,
#         num_experts: int,
#         topk: int,
#         dropout: float = 0.3,
#         router_hidden: int = 128,
#         router_dropout: float = 0.0,
#         gate_hidden: int = 64,
#         lb_lambda: float = 0.0
#     ):
#         super().__init__()
#         self.norm1 = nn.RMSNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

#         self.norm2 = nn.RMSNorm(d_model)
#         self.moe_ffn = MoEFeedForward(
#             d_model=d_model,
#             d_ff=d_ff,
#             num_experts=num_experts,
#             topk=topk,
#             router_hidden=router_hidden,
#             router_dropout=router_dropout,
#             gate_hidden=gate_hidden,
#             dropout=dropout,
#             lb_lambda=lb_lambda
#         )

#     def forward(self, x: torch.Tensor, log_q: torch.Tensor, attn_mask=None):
#         # (A) shared attention
#         x1 = self.norm1(x)
#         y = self.attn(x1, x1, x1, attn_mask=attn_mask)[0]
#         x = x + y

#         # (B) MoE-FFN
#         x2 = self.norm2(x)
#         f = self.moe_ffn(x2, log_q)  # [B,L,D]
#         x = x + f
#         return x


# # =========================================================
# # Main Model (Scheme C)
# #   (A) Embedding + pred tokens
# #   (B) Patch Intra/Inter with TransformerMoEBlock (MoE in FFN)
# #   (C) GMM prior weights directly from x channels -> log_q over TOTAL_LEN tokens
# #   (D) Final single prediction head on tail tokens
# #   (E) Expose aux_loss (optional) for load-balance
# # =========================================================
# class ThreeExpertPatchTransformer(nn.Module):
#     """
#     Scheme C: MoE in FFN layers.

#     - x already contains GMM responsibility weights (prior) in channels:
#         weight_seq = x[:, :, seq_start:seq_end]   # [B, seq_len, E]
#         weight_pt  = x[:, :, pt_start:pt_end]     # [B, seq_len, E]
#       ww_raw = weight_seq + seq_weight * weight_pt

#     - Build prior over total_len tokens:
#         prior_hist = normalize(ww_raw)                          -> [B, seq_len, E]
#         prior_pred = last pred_len of prior_hist (or repeat)    -> [B, pred_len, E]
#         prior_full = concat(prior_hist, prior_pred)             -> [B, total_len, E]
#         log_q_full = log(prior_full)

#     - Each Transformer block uses MoE-FFN:
#         gate_logits = FusionGate(log_q_full, RouterMLP(tokens))
#         Top-k -> combine expert FFN outputs
#     """

#     def __init__(
#         self,
#         seq_len: int,
#         pred_len: int,
#         patch_len: int,
#         d_model: int,
#         win_size: int,
#         revin: bool,
#         num_heads: int,
#         use_memory: bool,  # ignored; kept for signature compatibility
#         num_layers_intra_patch: int,
#         num_layers_inter_patch: int,
#         config=None,
#         c_in: int = 8,
#     ):
#         super().__init__()
#         self.config = config
#         self.revin = revin
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.total_len = seq_len + pred_len
#         self.d_model = d_model
#         self.c_in = c_in
#         self.patch_len = patch_len
#         self.win_size = win_size
#         self.num_heads = num_heads
#         self.num_layers_intra_patch = num_layers_intra_patch
#         self.num_layers_inter_patch = num_layers_inter_patch
#         self.eps = 1e-8

#         assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
#         self.num_patches = self.total_len // self.patch_len

#         # -------------------------
#         # (C) GMM weight slices from x
#         # -------------------------
#         self.seq_start = int(getattr(config, "gmm_seq_start", 5))
#         self.seq_end   = int(getattr(config, "gmm_seq_end", 8))
#         self.pt_start  = int(getattr(config, "gmm_pt_start", 2))
#         self.pt_end    = int(getattr(config, "gmm_pt_end", 5))
#         self.seq_weight = float(getattr(config, "seq_weight", 0.4))

#         self.num_experts = int(self.seq_end - self.seq_start)
#         assert self.num_experts > 0, "num_experts must be positive (check gmm_seq_start/end)."
#         assert (self.pt_end - self.pt_start) == self.num_experts, "pt slice dim must equal seq slice dim."
#         assert self.seq_end <= c_in and self.pt_end <= c_in, "x does not contain required GMM weight channels."

#         self.topk_experts = int(getattr(config, "topk_experts", 1))
#         self.topk_experts = max(1, min(self.topk_experts, self.num_experts))

#         # -------------------------
#         # (A) Embedding + pred tokens
#         # -------------------------
#         self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.3)
#         self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

#         # -------------------------
#         # MoE block hyperparams
#         # -------------------------
#         d_ff = int(getattr(config, "d_ff", d_model * 4))
#         router_hidden = int(getattr(config, "router_hidden", 128))
#         router_dropout = float(getattr(config, "router_dropout", 0.0))
#         gate_hidden = int(getattr(config, "gate_hidden", 64))
#         lb_lambda = float(getattr(config, "lb_lambda", 0.0))  # optional, default 0

#         # -------------------------
#         # (B) Shared backbone blocks but FFN is MoE
#         # -------------------------
#         self.intra = nn.ModuleList([
#             TransformerMoEBlock(
#                 d_model=d_model,
#                 num_heads=num_heads,
#                 d_ff=d_ff,
#                 num_experts=self.num_experts,
#                 topk=self.topk_experts,
#                 dropout=0.3,
#                 router_hidden=router_hidden,
#                 router_dropout=router_dropout,
#                 gate_hidden=gate_hidden,
#                 lb_lambda=lb_lambda
#             )
#             for _ in range(self.num_layers_intra_patch)
#         ])
#         self.inter = nn.ModuleList([
#             TransformerMoEBlock(
#                 d_model=d_model,
#                 num_heads=num_heads,
#                 d_ff=d_ff,
#                 num_experts=self.num_experts,
#                 topk=self.topk_experts,
#                 dropout=0.3,
#                 router_hidden=router_hidden,
#                 router_dropout=router_dropout,
#                 gate_hidden=gate_hidden,
#                 lb_lambda=lb_lambda
#             )
#             for _ in range(self.num_layers_inter_patch)
#         ])
#         self.post_norm = nn.RMSNorm(d_model)

#         # -------------------------
#         # (D) Final prediction head (single head)
#         # -------------------------
#         self.out_head = nn.Linear(d_model, 1)

#         # Expose aux loss (sum over blocks)
#         self.aux_loss = torch.tensor(0.0)

#     # =====================================================
#     # (C) Build GMM prior weights directly from x
#     #   return:
#     #     log_q_full: [B, total_len, E]
#     # =====================================================
#     def _build_log_q_full_from_x(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, seq_len, c_in]
#         Build token-wise prior probs for history tokens + pred tokens.
#         """
#         weight_seq = x[:, :, self.seq_start:self.seq_end]   # [B, seq_len, E]
#         weight_pt  = x[:, :, self.pt_start:self.pt_end]     # [B, seq_len, E]
#         ww_raw = weight_seq + self.seq_weight * weight_pt   # [B, seq_len, E]

#         # normalize history prior
#         ww_hist = ww_raw.clamp_min(self.eps)
#         ww_hist = ww_hist / (ww_hist.sum(dim=-1, keepdim=True) + self.eps)  # [B, seq_len, E]

#         # build pred prior by taking last pred_len (or repeat last)
#         if self.seq_len >= self.pred_len:
#             ww_pred = ww_hist[:, -self.pred_len:, :]  # [B, pred_len, E]
#         else:
#             pad = ww_hist[:, -1:, :].expand(x.size(0), self.pred_len - self.seq_len, self.num_experts)
#             ww_pred = torch.cat([ww_hist, pad], dim=1)

#         ww_full = torch.cat([ww_hist, ww_pred], dim=1)  # [B, total_len, E]
#         ww_full = ww_full.clamp_min(self.eps)
#         log_q_full = torch.log(ww_full)
#         return log_q_full

#     # =====================================================
#     # (B) Backbone forward with patching, passing log_q to MoE blocks
#     # =====================================================
#     def _forward_backbone(self, x_emb: torch.Tensor, log_q_full: torch.Tensor, intra_mask, inter_mask):
#         """
#         x_emb:      [B, total_len, d_model]
#         log_q_full: [B, total_len, E]
#         """
#         B = x_emb.size(0)

#         # Split to patches
#         patches = rearrange(x_emb, "b (p pl) d -> b p pl d", p=self.num_patches, pl=self.patch_len)
#         logq_p  = rearrange(log_q_full, "b (p pl) e -> b p pl e", p=self.num_patches, pl=self.patch_len)

#         # ---- Intra (within patch) ----
#         patches_intra = rearrange(patches, "b p pl d -> (b p) pl d").contiguous()
#         logq_intra    = rearrange(logq_p,  "b p pl e -> (b p) pl e").contiguous()

#         for block in self.intra:
#             patches_intra = block(patches_intra, log_q=logq_intra, attn_mask=intra_mask)

#         patches_intra = rearrange(patches_intra, "(b p) pl d -> b p pl d", b=B, p=self.num_patches).contiguous()

#         intra_tokens = rearrange(patches_intra, "b p pl d -> b (p pl) d")

#         # ---- Inter (across patches) ----
#         inter_patches = rearrange(patches_intra, "b p pl d -> (b pl) p d").contiguous()
#         logq_inter    = rearrange(logq_p,        "b p pl e -> (b pl) p e").contiguous()

#         for block in self.inter:
#             inter_patches = block(inter_patches, log_q=logq_inter, attn_mask=inter_mask)

#         inter_tokens = rearrange(inter_patches, "(b pl) p d -> b (p pl) d", b=B, pl=self.patch_len)

#         return self.post_norm(intra_tokens + inter_tokens)

#     # =====================================================
#     # Forward
#     # =====================================================
#     def forward(self, x: torch.Tensor, x_mark=None, y_true=None, sample_ids=None):
#         """
#         x: [B, seq_len, c_in] (must contain GMM weights channels)
#         return:
#           y: [B, pred_len, 1]
#         """
#         # (C) token-wise log prior for total_len tokens
#         log_q_full = self._build_log_q_full_from_x(x)  # [B, total_len, E]

#         # (A) embedding + pred tokens
#         x_emb_hist = self.embedding(x)                                # [B, seq_len, d_model]
#         B = x_emb_hist.size(0)
#         pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
#         x_emb = torch.cat([x_emb_hist, pred_token], dim=1)            # [B, total_len, d_model]

#         # masks
#         intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
#         inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

#         # (B) backbone with MoE-FFN blocks
#         final_shared = self._forward_backbone(x_emb, log_q_full, intra_mask, inter_mask)  # [B,total_len,d_model]

#         # (Optional) collect aux loss from all MoE blocks
#         aux = 0.0
#         if self.training:
#             for blk in self.intra:
#                 aux = aux + blk.moe_ffn.aux_loss
#             for blk in self.inter:
#                 aux = aux + blk.moe_ffn.aux_loss
#         self.aux_loss = aux if isinstance(aux, torch.Tensor) else torch.tensor(aux, device=x.device)

#         # (D) final head on tail tokens
#         y = self.out_head(final_shared[:, -self.pred_len:, :])  # [B,pred_len,1]
#         return y

#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer (Scheme C: MoE in FFN layers, Router uses RAW x, NO Memory)
#           - Attention is shared
#           - FFN is MoE-FFN (Top-k experts) inside each Transformer block
#           - Router logits are computed primarily from RAW x features (delta/prob/GMM in x channels)
#           - (Optional) token-router residual from hidden states
#           - Top-k select experts per token, weighted sum as FFN output
#           - Final prediction uses a SINGLE head on tail tokens

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.embedding import DataEmbedding
from modules.IFT_EncDec import ImplicitForecaster  # kept for compatibility; not used


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

    # additive mask: visible=0, masked=-inf
    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_bias.masked_fill_(upper, float("-inf"))
    return attn_bias


# =========================================================
# Module 1) Top-k sparse softmax on expert dimension
# =========================================================
def topk_sparse_softmax(logits: torch.Tensor, k: int, tau: torch.Tensor):
    """
    logits: [B, L, E]
    tau   : scalar tensor
    return:
      top_idx: [B, L, k]
      top_w  : [B, L, k]  (softmax over top-k)
    """
    B, L, E = logits.shape
    k = min(int(k), int(E))
    topv, topi = torch.topk(logits, k=k, dim=-1)  # [B, L, k]
    tau = tau.clamp_min(1e-6)
    topw = torch.softmax(topv / tau, dim=-1)
    return topi, topw


# =========================================================
# Module 2) Expert FFN (one expert)
# =========================================================
class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, L, d_model]


# =========================================================
# Module 3) Router that consumes RAW x directly
#   Input : x_full [B, total_len, c_in]
#   Output: router_logits_x [B, total_len, E]
# =========================================================
class RouterFromX(nn.Module):
    def __init__(self, c_in: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, x_full: torch.Tensor) -> torch.Tensor:
        return self.net(x_full)  # [B, L, E]


# =========================================================
# Module 4) Optional token-router residual (from hidden states)
#   Input : tokens [B, L, d_model]
#   Output: router_logits_h [B, L, E]
# =========================================================
class RouterFromHidden(nn.Module):
    def __init__(self, d_model: int, num_experts: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)  # [B, L, E]


# =========================================================
# Module 5) MoE-FFN with EXTERNAL router logits
#   - Experts: E separate FFNs
#   - Router logits come from Router(X) (and optional Router(hidden))
#   - Top-k select experts -> weighted sum = FFN output
# =========================================================
class MoEFeedForwardExternalRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        topk: int,
        dropout: float = 0.3,
        lb_lambda: float = 0.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.E = int(num_experts)
        self.topk = int(topk)
        self.lb_lambda = float(lb_lambda)
        self.eps = eps

        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout=dropout) for _ in range(self.E)])

        # trainable temperature
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

        # aux loss (load balance)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor):
        """
        x           : [B, L, d_model]
        router_logits: [B, L, E]
        """
        B, L, D = x.shape
        assert router_logits.shape == (B, L, self.E), "router_logits shape mismatch"

        tau = F.softplus(self.tau_raw) + 1e-3
        topi, topw = topk_sparse_softmax(router_logits, k=self.topk, tau=tau)  # [B,L,k], [B,L,k]

        # expert outputs: [B, L, E, D]
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=2)

        # gather selected: [B,L,k,D]
        idx = topi.unsqueeze(-1).expand(-1, -1, -1, D)
        sel = expert_outs.gather(dim=2, index=idx)

        # weighted sum: [B,L,D]
        y = (topw.unsqueeze(-1) * sel).sum(dim=2)

        # optional load-balance
        if self.lb_lambda > 0.0 and self.training:
            p = torch.softmax(router_logits, dim=-1)      # [B,L,E]
            mean_p = p.mean(dim=(0, 1))                   # [E]
            uniform = torch.full_like(mean_p, 1.0 / self.E)
            self.aux_loss = self.lb_lambda * ((mean_p - uniform) ** 2).sum()
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        return y


# =========================================================
# Module 6) Transformer block with MoE-FFN (router external)
#   - Attention shared
#   - FFN replaced by MoEFeedForwardExternalRouter
# =========================================================
class TransformerMoEBlockXRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        topk: int,
        dropout: float = 0.3,
        lb_lambda: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.RMSNorm(d_model)
        self.moe_ffn = MoEFeedForwardExternalRouter(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            topk=topk,
            dropout=dropout,
            lb_lambda=lb_lambda
        )

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor, attn_mask=None):
        # (A) shared attention
        x1 = self.norm1(x)
        y = self.attn(x1, x1, x1, attn_mask=attn_mask)[0]
        x = x + y

        # (B) MoE-FFN with external router logits
        x2 = self.norm2(x)
        f = self.moe_ffn(x2, router_logits)  # [B,L,D]
        x = x + f
        return x


# =========================================================
# Main Model (Scheme C, Router(X))
# =========================================================
class ThreeExpertPatchTransformer(nn.Module):
    """
    Scheme C: MoE in FFN layers, Router uses RAW x.

    Your x channel layout (as you described):
      - channel 0: differenced series (delta)
      - channel 1: probability vector (scalar prob)  (if it is a vector, you must allocate more channels)
      - channel 2:...: GMM responsibility weights (E dims)

    Default:
      gmm_start = 2, gmm_end = c_in  => E = c_in - 2

    Router uses full x (all channels) to output logits over E experts.
    Optional: add hidden-token router residual with coefficient alpha.
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
        use_memory: bool,  # ignored
        num_layers_intra_patch: int,
        num_layers_inter_patch: int,
        config=None,
        c_in: int = 8,
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
        self.num_heads = num_heads
        self.num_layers_intra_patch = num_layers_intra_patch
        self.num_layers_inter_patch = num_layers_inter_patch

        assert self.total_len % self.patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // self.patch_len

        # -------------------------
        # Expert count inferred from x's GMM weight channels
        # -------------------------
        self.gmm_start = int(getattr(config, "gmm_start", 2))
        self.gmm_end = int(getattr(config, "gmm_end", c_in))  # default: to last channel
        assert 0 <= self.gmm_start < self.gmm_end <= c_in, "Invalid gmm_start/gmm_end"
        self.num_experts = int(self.gmm_end - self.gmm_start)

        # Top-k
        self.topk_experts = int(getattr(config, "topk_experts", 1))
        self.topk_experts = max(1, min(self.topk_experts, self.num_experts))

        # How to build x features for pred tokens
        # "repeat_last" (default) or "zeros"
        self.pred_feat_mode = str(getattr(config, "pred_feat_mode", "repeat_last"))

        # -------------------------
        # Embedding + pred tokens
        # -------------------------
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.3)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        # -------------------------
        # Router(X): consumes raw x_full and outputs logits over experts
        # -------------------------
        router_x_hidden = int(getattr(config, "router_x_hidden", 128))
        router_x_dropout = float(getattr(config, "router_x_dropout", 0.0))
        self.router_x = RouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_x_hidden, dropout=router_x_dropout)

        # Optional token-router residual
        self.use_token_router = bool(getattr(config, "use_token_router", False))
        self.router_alpha = float(getattr(config, "router_alpha", 0.5))  # only used if use_token_router=True
        if self.use_token_router:
            router_h_hidden = int(getattr(config, "router_h_hidden", 128))
            router_h_dropout = float(getattr(config, "router_h_dropout", 0.0))
            self.router_h = RouterFromHidden(d_model=d_model, num_experts=self.num_experts, hidden=router_h_hidden, dropout=router_h_dropout)

        # -------------------------
        # MoE block hyperparams
        # -------------------------
        d_ff = int(getattr(config, "d_ff", d_model * 4))
        lb_lambda = float(getattr(config, "lb_lambda", 0.0))

        # -------------------------
        # Backbone blocks (Attention shared, FFN is MoE with external router)
        # -------------------------
        self.intra = nn.ModuleList([
            TransformerMoEBlockXRouter(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_experts=self.num_experts,
                topk=self.topk_experts,
                dropout=0.3,
                lb_lambda=lb_lambda
            )
            for _ in range(self.num_layers_intra_patch)
        ])
        self.inter = nn.ModuleList([
            TransformerMoEBlockXRouter(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                num_experts=self.num_experts,
                topk=self.topk_experts,
                dropout=0.3,
                lb_lambda=lb_lambda
            )
            for _ in range(self.num_layers_inter_patch)
        ])
        self.post_norm = nn.RMSNorm(d_model)

        # Output head (single head)
        self.out_head = nn.Linear(d_model, 1)

        # Expose aux loss
        self.aux_loss = torch.tensor(0.0)

    # =====================================================
    # Build x_full for router: [B, total_len, c_in]
    # =====================================================
    def _build_x_full_for_router(self, x_hist: torch.Tensor) -> torch.Tensor:
        """
        x_hist: [B, seq_len, c_in]
        return: x_full [B, total_len, c_in]
        """
        B = x_hist.size(0)
        if self.pred_feat_mode == "zeros":
            x_pred = torch.zeros(B, self.pred_len, self.c_in, device=x_hist.device, dtype=x_hist.dtype)
        else:
            # default: repeat_last
            last = x_hist[:, -1:, :]  # [B,1,c_in]
            x_pred = last.expand(B, self.pred_len, self.c_in).contiguous()

        x_full = torch.cat([x_hist, x_pred], dim=1)  # [B,total_len,c_in]
        return x_full

    # =====================================================
    # Backbone forward (patching) with external router logits
    # =====================================================
    def _forward_backbone(self, x_emb: torch.Tensor, router_logits_full: torch.Tensor, intra_mask, inter_mask):
        """
        x_emb:             [B, total_len, d_model]
        router_logits_full:[B, total_len, E]
        """
        B = x_emb.size(0)

        # Split to patches
        patches = rearrange(x_emb, "b (p pl) d -> b p pl d", p=self.num_patches, pl=self.patch_len)
        r_p = rearrange(router_logits_full, "b (p pl) e -> b p pl e", p=self.num_patches, pl=self.patch_len)

        # ---- Intra ----
        patches_intra = rearrange(patches, "b p pl d -> (b p) pl d").contiguous()
        r_intra = rearrange(r_p, "b p pl e -> (b p) pl e").contiguous()

        for block in self.intra:
            patches_intra = block(patches_intra, router_logits=r_intra, attn_mask=intra_mask)

        patches_intra = rearrange(patches_intra, "(b p) pl d -> b p pl d", b=B, p=self.num_patches).contiguous()
        intra_tokens = rearrange(patches_intra, "b p pl d -> b (p pl) d")

        # ---- Inter ----
        inter_patches = rearrange(patches_intra, "b p pl d -> (b pl) p d").contiguous()
        r_inter = rearrange(r_p, "b p pl e -> (b pl) p e").contiguous()

        for block in self.inter:
            inter_patches = block(inter_patches, router_logits=r_inter, attn_mask=inter_mask)

        inter_tokens = rearrange(inter_patches, "(b pl) p d -> b (p pl) d", b=B, pl=self.patch_len)

        return self.post_norm(intra_tokens + inter_tokens)

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x: torch.Tensor, x_mark=None, y_true=None, sample_ids=None):
        """
        x: [B, seq_len, c_in]
        return y: [B, pred_len, 1]
        """
        # 1) Build x_full for router logits (history + pred token features)
        x_full = self._build_x_full_for_router(x)                 # [B,total_len,c_in]

        # 2) Router logits from raw x
        #    Only route over GMM expert channels size E, but we let router see all channels of x.
        router_logits_full = self.router_x(x_full)                # [B,total_len,E]

        # 3) Embedding + pred tokens for backbone
        x_emb_hist = self.embedding(x)                            # [B,seq_len,d_model]
        B = x_emb_hist.size(0)
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)        # [B,total_len,d_model]

        # 4) Optional: add token-router residual (more adaptive)
        if self.use_token_router:
            # Use current embedding (before backbone) as an initial token signal for routing;
            # you can also move this to inside backbone if you want layer-wise updates later.
            router_logits_h = self.router_h(x_emb)                # [B,total_len,E]
            router_logits_full = router_logits_full + self.router_alpha * router_logits_h

        # 5) masks
        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # 6) Backbone with MoE-FFN blocks (external router)
        final_shared = self._forward_backbone(x_emb, router_logits_full, intra_mask, inter_mask)

        # 7) Collect aux loss (load-balance)
        aux = 0.0
        if self.training:
            for blk in self.intra:
                aux = aux + blk.moe_ffn.aux_loss
            for blk in self.inter:
                aux = aux + blk.moe_ffn.aux_loss
        self.aux_loss = aux if isinstance(aux, torch.Tensor) else torch.tensor(aux, device=x.device)

        # 8) Output head on tail tokens
        y = self.out_head(final_shared[:, -self.pred_len:, :])    # [B,pred_len,1]
        return y
