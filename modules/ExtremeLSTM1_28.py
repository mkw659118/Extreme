#Author  :   mkw
#Time    :   2025/09/17 10:50:52
#Desc    :   ExtremeLSTM (Turning-Point Memory 改进版：2-6点)
# 2) value 改 residual：存 (y_true - y_base)，推理加 residual
# 3) beta 门控加入：检索置信度 + Extreme 概率（router_prob[:, extreme_id]）
# 4) top-k 变硬：topk=1~2 + softmax 温度 + 相似度阈值
# 5) 事件定义更贴合差分：加入“幅值/跳变极端”触发，而非只看符号翻转
# 6) key 编码增强：加入窗口尾部形状（tail flatten）特征

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.embedding import DataEmbedding


# =========================================================
# Heads
# =========================================================
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
        self.proj = nn.Linear(hidden, 1)   # 保留 proj

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
        self.proj = nn.Linear(d_model, 1)         # 统一 proj: d_model -> 1

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        x = self.drop3(self.act(self.fc3(x)))
        return self.proj(x)


# =========================================================
# Turning Point / Extreme Event Score
# =========================================================
def _turning_score_from_diff(diff_1d: torch.Tensor,
                             eps: float = 0.05,
                             min_abs: float = 0.6,
                             min_jump: float = 0.6,
                             region_start: int = 0,
                             region_end: int = None):
    """
    turning point：符号翻转 + 幅值阈值 + 跳变阈值
    diff_1d: [B, T]
    return:
      has_tp:   [B] bool
      score_tp: [B] float  (越大越强)
    """
    B, T = diff_1d.shape
    if region_end is None:
        region_end = T - 1
    region_end = min(region_end, T - 1)

    # dead-zone 符号，抑制 0 附近的微振荡
    s = torch.sign(diff_1d)
    s = torch.where(diff_1d.abs() < eps, torch.zeros_like(s), s)

    # t-1 与 t 的符号翻转，定位到 t
    flip = (s[:, 1:] * s[:, :-1] < 0)  # [B, T-1]

    d0 = diff_1d[:, :-1]  # [B, T-1]
    d1 = diff_1d[:, 1:]   # [B, T-1]
    amp_ok = (d0.abs() > min_abs) & (d1.abs() > min_abs)
    jump_ok = ((d1 - d0).abs() > min_jump)

    valid = flip & amp_ok & jump_ok  # [B, T-1]

    # region mask on "t" in [1..T-1]
    t = torch.arange(1, T, device=diff_1d.device)  # [T-1]
    region = (t >= region_start) & (t <= region_end)
    valid = valid & region.view(1, -1)

    # turning 强度：幅值 + 跳变（只在 valid 位置）
    score_each = (d0.abs() + d1.abs() + (d1 - d0).abs())  # [B, T-1]
    score_each = torch.where(valid, score_each, torch.zeros_like(score_each))

    score, _ = score_each.max(dim=1)  # [B]
    has_tp = score > 0
    return has_tp, score


def _extreme_score_from_diff(diff_1d: torch.Tensor,
                             region_start: int = 0,
                             region_end: int = None):
    """
    更贴合“差分预测 RMSE”的极端事件刻画：
      - max_abs：幅值极端（尖峰）
      - max_jump：相邻差分跳变极端（jerk/突变）
    diff_1d: [B, T]
    return:
      max_abs:  [B]
      max_jump: [B]
    """
    B, T = diff_1d.shape
    if region_end is None:
        region_end = T - 1
    region_end = min(region_end, T - 1)
    region_start = max(0, region_start)

    seg = diff_1d[:, region_start:region_end + 1]  # [B, L]
    max_abs = seg.abs().amax(dim=1)

    if seg.size(1) >= 2:
        jump = (seg[:, 1:] - seg[:, :-1]).abs()
        max_jump = jump.amax(dim=1)
    else:
        max_jump = torch.zeros(B, device=diff_1d.device, dtype=diff_1d.dtype)

    return max_abs, max_jump


# =========================================================
# Key Encoder（增强：加入 tail 形状信息）
# =========================================================
class TurningPointKeyEncoder(nn.Module):
    """
    用窗口统计量 + 尾部形状（tail flatten）来编码 key
    """
    def __init__(self,
                 in_ch: int,
                 key_dim: int = 64,
                 hidden: int = 128,
                 dropout: float = 0.0,
                 tail_len: int = 8):
        super().__init__()
        self.in_ch = int(in_ch)
        self.tail_len = int(tail_len)

        # 统计量：mean/std/max_abs/last -> 4*in_ch
        # 形状：tail flatten -> tail_len*in_ch
        in_dim = 4 * self.in_ch + self.tail_len * self.in_ch

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, key_dim),
        )

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        """
        x_win: [B, key_len, in_ch]
        return key: [B, key_dim] (L2 normalize)
        """
        B, L, C = x_win.shape
        # 统计量
        mean = x_win.mean(dim=1)                               # [B, C]
        std = x_win.std(dim=1, unbiased=False)                 # [B, C]
        max_abs = x_win.abs().amax(dim=1)                      # [B, C]
        last = x_win[:, -1, :]                                 # [B, C]

        # 尾部形状（截取最后 tail_len 步）
        tlen = min(self.tail_len, L)
        tail = x_win[:, -tlen:, :].reshape(B, tlen * C)         # [B, tlen*C]
        # 若 L < tail_len，则补零到固定维度（保证网络输入维度一致）
        if tlen < self.tail_len:
            pad = torch.zeros(B, (self.tail_len - tlen) * C, device=x_win.device, dtype=x_win.dtype)
            tail = torch.cat([pad, tail], dim=-1)

        feat = torch.cat([mean, std, max_abs, last, tail], dim=-1)
        key = self.net(feat)
        key = F.normalize(key, dim=-1)
        return key


# =========================================================
# Memory Bank（改为存 residual：delta = y_true - y_base）
# =========================================================
class TurningPointMemoryBank(nn.Module):
    """
    Ring-buffer memory for turning-point / extreme patterns.
      keys:   [N, key_dim]
      values: [N, pred_len, 1]  (存 residual：y_true - y_base)
    """
    def __init__(self, mem_size: int, key_dim: int, pred_len: int, topk: int = 2):
        super().__init__()
        self.mem_size = int(mem_size)
        self.key_dim = int(key_dim)
        self.pred_len = int(pred_len)
        self.topk = int(topk)

        self.register_buffer("keys", torch.zeros(self.mem_size, self.key_dim))
        self.register_buffer("values", torch.zeros(self.mem_size, self.pred_len, 1))
        self.register_buffer("valid", torch.zeros(self.mem_size, dtype=torch.bool))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))  # 写指针

    @torch.no_grad()
    def add(self, key: torch.Tensor, value: torch.Tensor):
        """
        key:   [B, key_dim] (normalized)
        value: [B, pred_len, 1] (residual)
        """
        B = key.size(0)
        for i in range(B):
            p = int(self.ptr.item())
            self.keys[p].copy_(key[i])
            self.values[p].copy_(value[i])
            self.valid[p] = True
            self.ptr[0] = (p + 1) % self.mem_size

    def retrieve(self,
                 query_key: torch.Tensor,
                 sim_topk: int = None,
                 sim_temp: float = 0.2,
                 sim_thr: float = 0.0,
                 return_stats: bool = False):
        """
        query_key: [B, key_dim]
        return:
          delta_mem: [B, pred_len, 1]
          (可选 stats) max_sim/margin/entropy: [B]
        """
        B = query_key.size(0)

        if self.valid.sum() == 0:
            delta_mem = torch.zeros(B, self.pred_len, 1, device=query_key.device, dtype=query_key.dtype)
            if return_stats:
                z = torch.zeros(B, device=query_key.device, dtype=query_key.dtype)
                return delta_mem, z, z, z
            return delta_mem

        keys = self.keys[self.valid]     # [M, key_dim]
        vals = self.values[self.valid]   # [M, pred_len, 1]

        # 余弦相似度（keys/query_key 都已 normalize）
        sim = query_key @ keys.t()       # [B, M]

        # 变硬：topk 小 + softmax 温度小
        k = int(sim_topk) if sim_topk is not None else self.topk
        k = max(1, min(k, sim.size(-1)))

        top = torch.topk(sim, k=k, dim=-1)
        idx = top.indices                # [B, k]
        topv = top.values                # [B, k]

        # 置信度统计（用于门控）
        max_sim = topv[:, 0]  # [B]
        if k >= 2:
            margin = topv[:, 0] - topv[:, 1]
        else:
            margin = topv[:, 0]

        # 相似度阈值：不够像则直接返回 0（配合 beta 门控更稳）
        use_mask = (max_sim > sim_thr)

        # 计算权重：softmax(topv / temp)
        temp = max(float(sim_temp), 1e-6)
        w = torch.softmax(topv / temp, dim=-1)  # [B, k]

        # entropy（越小越“硬/确定”）
        if k >= 2:
            entropy = -(w * (w + 1e-12).log()).sum(dim=-1)  # [B]
        else:
            entropy = torch.zeros(B, device=query_key.device, dtype=query_key.dtype)

        # gather values: [B, k, pred_len, 1] -> 加权和 -> [B, pred_len, 1]
        v = vals[idx]
        delta_mem = (v * w.view(B, k, 1, 1)).sum(dim=1)

        # 对低相似样本置零（避免注入噪声）
        if use_mask is not None:
            delta_mem = delta_mem * use_mask.view(B, 1, 1).to(delta_mem.dtype)

        if return_stats:
            return delta_mem, max_sim, margin, entropy
        return delta_mem


# =========================================================
# Router
# =========================================================
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
        std = x.std(dim=1, unbiased=False)       # [B, C]
        max_abs = x.abs().amax(dim=1)            # [B, C]
        last = x[:, -1, :]                       # [B, C]
        feat = torch.cat([std, max_abs, last], dim=-1)  # [B, 3C]
        return self.net(feat)


# =========================================================
# ExtremeLSTM（改 2-6 点）
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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in

        # -------- expert definition --------
        self.num_experts = 3
        self.extreme_expert_id = 2  # ExtremeHead 在 expert_heads 中的下标（0:Normal,1:Mid,2:Extreme）

        # -------- Embedding + pred tokens --------
        dropout = 0.3
        self.embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=dropout)
        self.pred_tokens = nn.Parameter(torch.randn(self.pred_len, self.d_model))

        enc_layers = int(num_layers_intra_patch)
        dec_layers = int(num_layers_inter_patch)

        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=enc_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.post_norm = nn.RMSNorm(d_model)

        # -------- router --------
        router_hidden = self.d_model
        router_dropout = dropout
        self.router = SampleRouterFromX(c_in=c_in, num_experts=self.num_experts, hidden=router_hidden, dropout=router_dropout)

        # -------- heads --------
        self.expert_heads = nn.ModuleList([
            NormalHead(d_model),
            MidHead(d_model, hidden=d_model, dropout=dropout),
            ExtremeHead(d_model, hidden=2 * d_model, dropout=dropout),
        ])

        # -------- top-k gating (MoE) --------
        self.top_k = 2

        # -------- GMM label slices (保留原逻辑，不动) --------
        self.gmm_pt_start  = 2
        self.gmm_pt_end    = 5
        self.gmm_seq_start = 7
        self.gmm_seq_end   = 10

        # =========================================================
        # Turning-Point / Extreme Memory Config
        # =========================================================
        self.use_memory = use_memory

        # 目标差分通道（你已确认没问题，这里保持你的设置逻辑）
        self.tp_target_idx = int(getattr(config, "tp_target_idx", 0))

        # key 截断长度
        self.tp_key_len = int(getattr(config, "tp_key_len", min(32, self.seq_len)))

        # 4) 变硬：默认 topk=2（建议 1~2）
        self.tp_topk = int(getattr(config, "tp_topk", 2))

        # 4) softmax 温度（越小越硬）
        self.tp_sim_softmax_temp = float(getattr(config, "tp_sim_softmax_temp", 0.15))

        # 4) 相似度阈值：低于则不注入
        self.tp_sim_thr = float(getattr(config, "tp_sim_thr", 0.35))

        # 3) margin/entropy 门控阈值（可选，默认比较宽松）
        self.tp_margin_thr = float(getattr(config, "tp_margin_thr", 0.05))
        self.tp_entropy_thr = float(getattr(config, "tp_entropy_thr", 0.85))  # k=2 时 entropy 最大约 ln2=0.693；这里做宽松上限
        self.tp_gate_temp = float(getattr(config, "tp_gate_temp", 0.25))      # 门控 sigmoid 温度（越小越“硬”）

        # 3) extreme 概率门控（router_prob[:, extreme_id]）
        self.tp_p_ext_thr = float(getattr(config, "tp_p_ext_thr", 0.33))
        self.tp_p_ext_temp = float(getattr(config, "tp_p_ext_temp", 0.20))

        # 记忆注入系数上限
        self.tp_beta = float(getattr(config, "tp_beta", 0.3))

        # turning point 检测参数
        self.tp_eps = float(getattr(config, "tp_eps", 0.05))
        self.tp_min_abs = float(getattr(config, "tp_min_abs", 0.6))
        self.tp_min_jump = float(getattr(config, "tp_min_jump", 0.6))
        self.tp_future_region = int(getattr(config, "tp_future_region", self.pred_len))

        # turning score 门控参数
        self.tp_score_thr = float(getattr(config, "tp_score_thr", 1.2))
        self.tp_score_temp = float(getattr(config, "tp_score_temp", 0.5))

        # 5) 幅值/跳变极端门控阈值（更贴合差分 RMSE）
        self.tp_hist_abs_thr = float(getattr(config, "tp_hist_abs_thr", 1.2))
        self.tp_hist_jump_thr = float(getattr(config, "tp_hist_jump_thr", 1.2))
        self.tp_abs_temp = float(getattr(config, "tp_abs_temp", 0.5))
        self.tp_jump_temp = float(getattr(config, "tp_jump_temp", 0.5))

        self.tp_fut_abs_thr = float(getattr(config, "tp_fut_abs_thr", 1.5))
        self.tp_fut_jump_thr = float(getattr(config, "tp_fut_jump_thr", 1.5))

        if self.use_memory:
            key_dim = int(getattr(config, "tp_key_dim", 64))
            key_hidden = int(getattr(config, "tp_key_hidden", 128))
            key_drop = float(getattr(config, "tp_key_dropout", 0.0))
            mem_size = int(getattr(config, "tp_mem_size", 4096))
            tail_len = int(getattr(config, "tp_tail_len", 8))

            # 6) key 编码增强：tail flatten
            self.tp_key_encoder = TurningPointKeyEncoder(
                in_ch=1,
                key_dim=key_dim,
                hidden=key_hidden,
                dropout=key_drop,
                tail_len=tail_len
            )

            # 2) 存 residual
            self.tp_memory = TurningPointMemoryBank(
                mem_size=mem_size,
                key_dim=key_dim,
                pred_len=self.pred_len,
                topk=self.tp_topk
            )

    def _sigmoid_gate(self, x: torch.Tensor, thr: float, temp: float):
        """通用 sigmoid 门控：sigmoid((x - thr)/temp)"""
        t = max(float(temp), 1e-6)
        return torch.sigmoid((x - float(thr)) / t)

    def forward(self, x, x_mark=None, y_true=None, sample_ids=None, route_labels=None):
        """
        x: [B, seq_len, c_in]
        return y: [B, pred_len, 1]
        """
        B = x.size(0)

        # ---------------- routing ----------------
        router_logits = self.router(x)                         # [B, E]
        router_prob = torch.softmax(router_logits, dim=-1)      # [B, E]
        p_extreme = router_prob[:, self.extreme_expert_id]      # [B]  Extreme 概率（用于门控）

        # ---------------- embedding ----------------
        x_emb_hist = self.embedding(x)                          # [B, seq_len, d_model]

        # ---------------- LSTM backbone ----------------
        _, (h_n, c_n) = self.encoder(x_emb_hist)                # [L, B, d_model]
        pred_token = self.pred_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, pred_len, d_model]
        dec_out, _ = self.decoder(pred_token, (h_n, c_n))       # [B, pred_len, d_model]
        final_shared = self.post_norm(dec_out)                  # [B, pred_len, d_model]

        # ---------------- MoE heads ----------------
        expert_preds = torch.cat([head(final_shared) for head in self.expert_heads], dim=-1)  # [B, pred_len, E]

        # top-k experts per sample
        k = self.top_k
        topk_result = torch.topk(router_prob, k=k, dim=-1)
        topk_probs = topk_result.values       # [B, k]
        topk_experts = topk_result.indices    # [B, k]

        # top-k 内归一化
        mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # [B, k]

        # gather expert outputs
        expert_index = topk_experts[:, None, :].expand(B, self.pred_len, k)       # [B, pred_len, k]
        chosen_expert_preds = expert_preds.gather(dim=-1, index=expert_index)     # [B, pred_len, k]

        mix_weights = mix_weights[:, None, :].expand(B, self.pred_len, k)         # [B, pred_len, k]

        # ---------------- base prediction (不带记忆) ----------------
        y_base = (chosen_expert_preds * mix_weights).sum(dim=-1, keepdim=True)    # [B, pred_len, 1]
        y = y_base

        # =========================================================
        # Memory (2-6点改进)
        # =========================================================
        if self.use_memory:
            # 取目标差分列，截断 key_len
            x_tgt = x[:, :, self.tp_target_idx:self.tp_target_idx + 1]            # [B, seq_len, 1]
            x_win = x_tgt[:, -self.tp_key_len:, :]                                # [B, key_len, 1]
            q_key = self.tp_key_encoder(x_win)                                    # [B, key_dim]

            # 4) top-k 变硬 + softmax 温度 + 相似度阈值
            delta_mem, max_sim, margin, entropy = self.tp_memory.retrieve(
                q_key,
                sim_topk=self.tp_topk,
                sim_temp=self.tp_sim_softmax_temp,
                sim_thr=self.tp_sim_thr,
                return_stats=True
            )  # delta_mem: [B, pred_len, 1]

            # 5) 历史事件刻画：turning + 幅值/跳变极端
            has_hist_tp, hist_tp_score = _turning_score_from_diff(
                x_win.squeeze(-1),
                eps=self.tp_eps,
                min_abs=self.tp_min_abs,
                min_jump=self.tp_min_jump,
                region_start=max(1, self.tp_key_len - 8),
                region_end=self.tp_key_len - 1
            )

            hist_abs, hist_jump = _extreme_score_from_diff(
                x_win.squeeze(-1),
                region_start=max(0, self.tp_key_len - 8),
                region_end=self.tp_key_len - 1
            )

            # 3) 门控：turning 强度门控 + 幅值/跳变门控
            gate_tp = self._sigmoid_gate(hist_tp_score, self.tp_score_thr, self.tp_score_temp)
            gate_abs = self._sigmoid_gate(hist_abs, self.tp_hist_abs_thr, self.tp_abs_temp)
            gate_jump = self._sigmoid_gate(hist_jump, self.tp_hist_jump_thr, self.tp_jump_temp)

            # 3) 门控：检索置信度（max_sim/margin/entropy）
            gate_sim = self._sigmoid_gate(max_sim, self.tp_sim_thr, self.tp_gate_temp)
            gate_margin = self._sigmoid_gate(margin, self.tp_margin_thr, self.tp_gate_temp)
            # entropy 越小越好：用 (thr - entropy)
            gate_entropy = torch.sigmoid((float(self.tp_entropy_thr) - entropy) / max(self.tp_gate_temp, 1e-6))

            # 3) 门控：Extreme 概率（尽量只在 extreme 场景开门）
            gate_regime = self._sigmoid_gate(p_extreme, self.tp_p_ext_thr, self.tp_p_ext_temp)

            # 组合门控（你可以按需调整乘法结构）
            # - gate_tp 捕获转折
            # - gate_abs/jump 捕获差分尖峰/突变
            # - gate_sim/margin/entropy 确保检索靠谱
            # - gate_regime 确保只在极端专家占比高时注入
            gate_event = torch.maximum(gate_tp, torch.maximum(gate_abs, gate_jump))  # 事件门控：转折 或 幅值/跳变极端
            beta = self.tp_beta * gate_event * gate_sim * gate_margin * gate_entropy * gate_regime  # [B]

            # 2) 融合：y = y_base + beta * delta_mem
            y = y_base + beta.view(B, 1, 1) * delta_mem

            # ---------------- 训练写入记忆：存 residual ----------------
            if self.training and (y_true is not None):
                # 你已确认通道一致性没问题，这里保持你原先的 y_true[:, :, 0:1] 选择方式
                y_tgt = y_true[:, :, 0:1]  # [B, pred_len, 1] （差分 label）

                # 2) residual：delta = y_true - y_base
                delta_true = (y_tgt - y_base).detach()

                # 未来事件：turning + 幅值/跳变极端（用真实未来差分来决定写入）
                d_all = torch.cat([x_tgt.squeeze(-1), y_tgt.squeeze(-1)], dim=1)  # [B, seq_len + pred_len]

                region_start = self.seq_len
                region_end = self.seq_len + min(self.tp_future_region, self.pred_len) - 1

                has_fut_tp, fut_tp_score = _turning_score_from_diff(
                    d_all,
                    eps=self.tp_eps,
                    min_abs=self.tp_min_abs,
                    min_jump=self.tp_min_jump,
                    region_start=region_start,
                    region_end=region_end
                )

                fut_abs, fut_jump = _extreme_score_from_diff(
                    d_all,
                    region_start=region_start,
                    region_end=region_end
                )

                # 5) 写入条件：强 turning 或 幅值/跳变极端
                write_mask = (
                    (has_fut_tp & (fut_tp_score > self.tp_score_thr)) |
                    (fut_abs > self.tp_fut_abs_thr) |
                    (fut_jump > self.tp_fut_jump_thr)
                )

                if write_mask.any():
                    # 只写入 residual（更稳，不会直接注入别人的未来轨迹）
                    self.tp_memory.add(q_key[write_mask].detach(), delta_true[write_mask])

        return y
