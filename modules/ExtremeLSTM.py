#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTM
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from layers.embedding import DataEmbedding
# 残差记忆库，全存，容量大小1024

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
        self.fc3 = nn.Linear(hidden, d_model)     # 压回 d_model
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)         # 所有专家统一 proj: d_model -> 1

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return self.proj(x)


def _turning_score_from_diff(diff_1d: torch.Tensor,
                             eps: float = 0.05,
                             min_abs: float = 0.6,
                             min_jump: float = 0.6,
                             region_start: int = 0,
                             region_end: int = None):
   
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
    def add(self, key: torch.Tensor, residual: torch.Tensor):
        """
        key:   [B, key_dim] (normalized)
        value: [B, pred_len, 1]
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
        r_mem = (v * w.view(-1, k, 1, 1)).sum(dim=1)
        return r_mem


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
        std = x.std(dim=1, unbiased=False)  # 1) std: [B, C]
        max_abs = x.abs().amax(dim=1)  # 2) max_abs: [B, C]
        last = x[:, -1, :]  # 3) last: [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)
    
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V):
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))  # [B,P,T]
        attn = F.softmax(scores, dim=-1)
        ctx  = torch.matmul(attn, v)  # [B,P,H]
        return ctx, attn  

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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dropout = self.config.dropout

        # -------- expert definition --------
        self.num_experts = 3
        
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
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.post_norm = nn.RMSNorm(d_model)
        self.xattn = CrossAttention(d_model)

        # -------- router --------
        router_hidden = self.d_model
        router_dropout = self.dropout
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- heads --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=self.dropout),
            ExtremeHead(d_model, hidden=2 * d_model, dropout=self.dropout),
        ])
        
        self.fuse_proj = nn.Linear(2 * d_model, d_model)


        # -------- top-k gating --------
        self.top_k = 2
        
        # -------- GMM label slices --------
        self.gmm_pt_start  = 2
        self.gmm_pt_end    = 5
        self.gmm_seq_start = 7
        self.gmm_seq_end   = 10

        # =========================================================
        # Turning-Point Memory 
        # =========================================================
        self.use_memory = use_memory
        self.tp_target_idx = int(getattr(config, "tp_target_idx", 0))
        self.tp_key_len = int(getattr(config, "tp_key_len", min(32, self.seq_len)))
        self.tp_topk = int(getattr(config, "tp_topk", 20))
        self.tp_beta = float(getattr(config, "tp_beta", 0.3))

        self.tp_eps = float(getattr(config, "tp_eps", 0.05))
        self.tp_min_abs = float(getattr(config, "tp_min_abs", 0.6))
        self.tp_min_jump = float(getattr(config, "tp_min_jump", 0.6))
        self.tp_future_region = int(getattr(config, "tp_future_region", self.pred_len))

        self.tp_score_thr = float(getattr(config, "tp_score_thr", 1.2))
        self.tp_score_temp = float(getattr(config, "tp_score_temp", 0.5))

        if self.use_memory:
            key_dim = int(getattr(config, "tp_key_dim", 64))
            key_hidden = int(getattr(config, "tp_key_hidden", 128))
            key_drop = float(getattr(config, "tp_key_dropout", 0.0))
            mem_size = int(getattr(config, "mem_size", 1024))

            self.tp_key_encoder = TurningPointKeyEncoder(in_ch=1, key_dim=key_dim, hidden=key_hidden, dropout=key_drop)
            self.tp_memory = TurningPointMemoryBank(mem_size=mem_size, key_dim=key_dim, pred_len=self.pred_len, topk=self.tp_topk)

    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        return y: [B, pred_len, 1]
        """
        B = x.size(0)
        # ---------------- routing ----------------
        router_logits = self.router(x)                       # [B, E]
        router_prob = torch.softmax(router_logits, dim=-1)    # [B, E]

        # ---------------- embedding ----------------
        x_emb_hist = self.embedding(x)                        # [B, seq_len, d_model]

        # =========================================================
        # LSTM backbone: encoder history -> decoder pred_tokens
        # =========================================================
        enc_out, (h_n, c_n) = self.encoder(x_emb_hist)              # h_n/c_n: [enc_layers, B, d_model]

        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        dec_out, _ = self.decoder(pred_token, (h_n, c_n))             # [B, pred_len, d_model]
        ctx, _ = self.xattn(dec_out, enc_out, enc_out)
        fused = torch.cat([dec_out, ctx], dim=-1)
        fused = self.fuse_proj(fused)          # [B, pred_len, d_model]

        final_shared = self.post_norm(fused)                        # [B, pred_len, d_model]

        # 1) 专家头：计算每个 expert head 的输出并在最后一维拼接
        # expert_preds: [B, pred_len, E]，E 为专家数
        expert_preds = torch.cat([head(final_shared) for head in self.expert_heads], dim=-1)

        # 2) 路由选择：对每个样本，从 router_prob 中选出概率最大的 top-k 个专家
        k = self.top_k
        topk_result = torch.topk(router_prob, k=k, dim=-1)

        topk_probs = topk_result.values     # [B, k]，top-k 专家的路由概率（尚未在 top-k 内归一化）
        topk_experts = topk_result.indices  # [B, k]，top-k 专家的编号（expert id）

        # 3) 权重归一化：只在 top-k 范围内做归一化，使每个样本的 top-k 权重和为 1
        mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

        # 4) 收集对应专家输出：把每个样本选中的 top-k 专家输出从 expert_preds 中 gather 出来
        # chosen_expert_preds: [B, pred_len, k]
        expert_index = topk_experts[:, None, :].expand(B, self.pred_len, k)       # [B, pred_len, k]，把索引扩展到每个预测步
        chosen_expert_preds = expert_preds.gather(dim=-1, index=expert_index)     # [B, pred_len, k]，取出 top-k 专家的预测

        # 5) 加权融合：用 top-k 权重对对应专家输出加权求和，得到最终预测
        mix_weights = mix_weights[:, None, :].expand(B, self.pred_len, k)         # [B, pred_len, k]，把权重扩展到每个预测步
        base = (chosen_expert_preds * mix_weights).sum(dim=-1, keepdim=True)         # [B, pred_len, 1]，最终预测

        if self.use_memory:
            # 取目标差分列，截断长度，编码成查询键向量
            x_tgt = x[:, :, self.tp_target_idx:self.tp_target_idx + 1]     # [B, seq_len, 1]
            x_win = x_tgt[:, -self.tp_key_len:, :]                         # [B, key_len, 1]
            q_key = self.tp_key_encoder(x_win)                             # [B, key_dim]
            # 做topK相似度检索返回
            r_mem = self.tp_memory.retrieve(q_key)                         # [B, pred_len, 1]
            # 获取拐点发生概率与数值强度分数
            has_hist_tp, hist_score = _turning_score_from_diff(
                x_win.squeeze(-1),
                eps=self.tp_eps,
                min_abs=self.tp_min_abs,
                min_jump=self.tp_min_jump,
                region_start=max(1, self.tp_key_len - 8),
                region_end=self.tp_key_len - 1
            )
            # 根据拐点强度计算模型修正系数
            beta = self.tp_beta * torch.sigmoid((hist_score - self.tp_score_thr) / max(self.tp_score_temp, 1e-6))
            y = base + beta.view(B, 1, 1) * r_mem
            # # 训练阶段写入记忆
            # 训练阶段：全存（不做拐点/极端值筛选），由 mem_size 控制只保留最近样本
            if self.training and (y_true is not None):
                y_tgt = y_true[:, :, 0:1]          # [B, pred_len, 1]
                res = (y_tgt - base.detach())         # [B, pred_len, 1]  存 residual
                self.tp_memory.add(q_key.detach(), res.detach())

        return y 