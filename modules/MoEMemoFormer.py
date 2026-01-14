#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   PatchExtremeMemoryTransformer
#           Shared Backbone + Top1 Expert Heads (sample-level routing)
#           Router uses RAW x (delta/prob/GMM) to select expert (Top1)
#           Anti-collapse constraints:
#             - Router supervised by GMM argmax label (distribution -> expert)
#             - Load-balance regularization
#             - Head diversity regularization (orthogonality)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers.embedding import DataEmbedding


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
# Module 1) Standard Transformer Block (NO MoE in FFN)
# =========================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.3):
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
#   - pool over time -> [B, c_in] summary
#   - output logits over experts: [B, E]
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
        x_hist: [B, seq_len, c_in]
        returns logits: [B, E]
        """
        # 你可以替换成更强的 pool：mean/std/max 等；这里先给最稳的 mean
        feat = x.mean(dim=1)         # [B, c_in]
        return self.net(feat)             # [B, E]


# =========================================================
# Module 3) Expert Heads (E heads)
#   - input: tail tokens [B, pred_len, d_model]
#   - output: y_e [B, pred_len, 1] for each expert e
# =========================================================
class ExpertHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, tail_tokens: torch.Tensor):
        return self.proj(tail_tokens)  # [B, pred_len, 1]


# =========================================================
# Module 4) Diversity regularizer for heads (prevent "too similar")
#   - orthogonality penalty between head weight vectors
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
    W = torch.stack(W, dim=0)       # [E, d_model]
    # cosine matrix: [E,E]
    C = W @ W.t()
    E = C.size(0)
    # remove diagonal
    off = C - torch.eye(E, device=C.device, dtype=C.dtype)
    return (off ** 2).mean()


# =========================================================
# Main Model
#   Shared Backbone + Top1 Expert Head
# =========================================================
class ThreeExpertPatchTransformer(nn.Module):
    """
    Shared backbone; experts only in prediction heads; router selects top-1 expert per sample.

    x layout (default):
      x[..., 0]   = delta
      x[..., 1]   = prob
      x[..., 2:]  = GMM responsibilities (E dims)

    Anti-collapse:
      - router_ce: router logits supervised by GMM argmax label (distribution -> expert)
      - load_balance: mean softmax probs close to uniform
      - head_div: orthogonality between head weights
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
        c_in: int = 9,
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

        # -------- expert definition from x's GMM channel range --------
        self.gmm_start = int(getattr(config, "gmm_start", 2))
        self.gmm_end   = int(getattr(config, "gmm_end", 5))
        assert 0 <= self.gmm_start < self.gmm_end <= c_in, "Invalid gmm_start/gmm_end"
        self.num_experts = int(self.gmm_end - self.gmm_start)

        # -------- losses weights (important for stability) --------
        self.w_router_ce = float(getattr(config, "w_router_ce", 1.0))      # router supervised by distribution label
        self.w_balance   = float(getattr(config, "w_balance", 0.01))       # load-balance
        self.w_head_div  = float(getattr(config, "w_head_div", 0.01))      # head diversity

        # if True: training uses teacher label (GMM argmax) to pick expert; inference uses router
        self.teacher_forcing = bool(getattr(config, "teacher_forcing", False))

        # -------- Embedding + pred tokens --------
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.3)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        # -------- Shared backbone blocks (NO MoE) --------
        d_ff = int(getattr(config, "d_ff", d_model * 4))
        dropout = float(getattr(config, "dropout", 0.3))

        self.intra = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_intra_patch)
        ])
        self.inter = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers_inter_patch)
        ])
        self.post_norm = nn.RMSNorm(d_model)

        # -------- Router (sample-level) --------
        router_hidden = int(getattr(config, "router_x_hidden", 256))
        router_dropout = float(getattr(config, "router_x_dropout", 0.3))
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- Expert heads --------
        self.expert_heads = nn.ModuleList([ExpertHead(d_model) for _ in range(self.num_experts)])

        # -------- Expose aux losses --------
        self.aux_loss_dict = {}

    # =====================================================
    # Build distribution label from GMM responsibilities in x (hard assignment)
    # =====================================================
    def _gmm_argmax_label(self, x):
        """
        x: [B, seq_len, c_in]
        label: [B] in {0..E-1}
        """
        weight_seq = x[:, :, 6:9]            # [B, seq_len, 3]
        weight_pt = x[:, :, 2:5]             # [B, seq_len, 3]
        ww = weight_seq + 0.4 * weight_pt
        gmm = ww
        # mean responsibilities over time -> distribution id
        gmm_mean = gmm.mean(dim=1)                  # [B, E]
        label = torch.argmax(gmm_mean, dim=-1)      # [B]
        return label

    # =====================================================
    # Shared backbone forward (patch intra + inter)
    # =====================================================
    def _forward_backbone(self, x_emb: torch.Tensor, intra_mask, inter_mask):
        B = x_emb.size(0)

        patches = rearrange(x_emb, "b (p pl) d -> b p pl d", p=self.num_patches, pl=self.patch_len)

        # intra: (B,P)->(B*P)
        patches_intra = rearrange(patches, "b p pl d -> (b p) pl d").contiguous()
        for blk in self.intra:
            patches_intra = blk(patches_intra, attn_mask=intra_mask)
        patches_intra = rearrange(patches_intra, "(b p) pl d -> b p pl d", b=B, p=self.num_patches).contiguous()

        intra_tokens = rearrange(patches_intra, "b p pl d -> b (p pl) d")

        # inter: [B,P,pl,d]->[B*pl,P,d]
        inter_patches = rearrange(patches_intra, "b p pl d -> (b pl) p d").contiguous()
        for blk in self.inter:
            inter_patches = blk(inter_patches, attn_mask=inter_mask)
        inter_tokens = rearrange(inter_patches, "(b pl) p d -> b (p pl) d", b=B, pl=self.patch_len)

        return self.post_norm(intra_tokens + inter_tokens)

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        route_labels (optional): [B] ground-truth routing label (0..E-1)
                               If None, we use GMM-argmax label as pseudo label.
        return:
          y: [B, pred_len, 1]
        """
        B = x.size(0)

        # 根据GMM和原始输入得到专家路由概率
        router_logits = self.router(x)                # [B, E]
        router_prob = torch.softmax(router_logits, dim=-1)  # [B, E]

        # ----- routing label (distribution -> expert) -----
        if route_labels is None:
            route_labels = self._gmm_argmax_label(x)  # [B]

        # ----- choose expert index -----
        if self.training and self.teacher_forcing:
            # training: force each distribution to its expert (stabilize early)
            expert_idx = route_labels                 # [B]
        else:
            # use router decision
            expert_idx = torch.argmax(router_logits, dim=-1)  # [B]

        # ----- embedding + pred tokens -----
        x_emb_hist = self.embedding(x)                                # [B, seq_len, d_model]
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        x_emb = torch.cat([x_emb_hist, pred_token], dim=1)            # [B, total_len, d_model]

        intra_mask = generate_causal_window_mask(self.patch_len, self.win_size, x_emb.device, x_emb.dtype)
        inter_mask = generate_causal_window_mask(self.num_patches, self.num_patches, x_emb.device, x_emb.dtype)

        # ----- shared backbone -----
        final_shared = self._forward_backbone(x_emb, intra_mask, inter_mask)    # [B, total_len, d_model]
        tail = final_shared[:, -self.pred_len:, :]                               # [B, pred_len, d_model]

        # ----- compute ALL head outputs (E is small => cheap), then gather Top1 -----
        # y_all: [B, pred_len, E]
        y_all = torch.cat([h(tail) for h in self.expert_heads], dim=-1)

        # gather selected: index shape [B, pred_len, 1]
        idx = expert_idx.view(B, 1, 1).expand(B, self.pred_len, 1)
        y = y_all.gather(dim=-1, index=idx)  # [B, pred_len, 1]

        # =====================================================
        # Aux losses (prevent collapse)
        # =====================================================
        aux = {}

        # (1) Router supervised by distribution label (GMM argmax or provided labels)
        if self.w_router_ce > 0.0 and self.training:
            aux["router_ce"] = self.w_router_ce * F.cross_entropy(router_logits, route_labels)

        # (2) Load balance: mean router_prob close to uniform (prevents always choosing one expert)
        if self.w_balance > 0.0 and self.training:
            mean_p = router_prob.mean(dim=0)  # [E]
            uniform = torch.full_like(mean_p, 1.0 / self.num_experts)
            aux["balance"] = self.w_balance * ((mean_p - uniform) ** 2).sum()

        # (3) Head diversity: orthogonality of head weights (prevents heads becoming too similar)
        if self.w_head_div > 0.0 and self.training and self.num_experts > 1:
            aux["head_div"] = self.w_head_div * head_diversity_loss(self.expert_heads)

        self.aux_loss_dict = aux

        return y
