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


class TinyBottleneckMemory(nn.Module):
    """
    超轻瓶颈记忆：
    - 仅在 r 维（远小于 d_model）维护 3 个原型；
    - 上/下投影用固定正交矩阵（buffer，不训练）；
    - 读：按步混合 -> [B, L_pred, d]；
    - 写：EMA（no_grad），带轻量正交化；
    - 另提供按步置信度门控（基于 ww 熵）。
    """

    def __init__(self, d_model: int, r: int = 8, momentum: float = 0.05, init_std: float = 1e-3):
        """
        初始化 TinyBottleneckMemory。

        参数：
        - d_model: 输入/输出特征维度 d
        - r: 瓶颈维度（r << d）
        - momentum: 写入时 EMA 的动量系数
        - init_std: 原型初始化的标准差

        初始化内容：
        1) 构造固定正交投影矩阵 P ∈ R^{d×r}（buffer，不训练），用于 d→r 与 r→d 投影；
        2) 初始化 3 个 r 维原型 z_protos（单位范数），作为瓶颈空间的记忆原型；
        3) 初始化 seen 计数器，记录每个原型累计“被使用/写入”的权重总和。
        """
        super().__init__()
        self.d_model = d_model
        self.r = r
        self.momentum = momentum

        with torch.no_grad():
            A = torch.randn(d_model, r)
            Q, _ = torch.linalg.qr(A, mode='reduced')  # Q: [d, r]
        self.register_buffer("P", Q)  # 固定正交投影（不参与训练）

        z = torch.randn(3, r) * init_std
        z = F.normalize(z, dim=-1)
        self.register_buffer("z_protos", z)           # 3 个瓶颈原型（不参与梯度）
        self.register_buffer("seen", torch.zeros(3))  # 记录原型使用强度（不参与梯度）

    @torch.no_grad()
    def _orthogonalize_(self):
        """
        对 3 个原型在 r 维空间做轻量 Gram-Schmidt 正交化（就地更新 z_protos）。

        目的：
        - 降低原型之间的冗余/塌缩，提高原型多样性；
        - 每次写入后执行一次，成本很低（仅 3 个向量）。

        实现：
        - 依次对每个原型减去在之前原型方向上的投影；
        - 再对每个原型做 L2 归一化；
        - 最终 copy_ 回 z_protos。
        """
        Z = self.z_protos.detach().clone()  # [3, r]
        for i in range(Z.size(0)):
            for j in range(i):
                coef = torch.dot(Z[i], Z[j])
                Z[i] = Z[i] - coef * Z[j]
            Z[i] = F.normalize(Z[i], dim=-1)
        self.z_protos.copy_(Z)

    @torch.no_grad()
    def write(self, ww: torch.Tensor, q_hist: torch.Tensor):
        """
        写入记忆：根据 ww（每步 3 原型权重）与 q_hist（历史摘要向量）更新瓶颈原型 z_protos。

        输入：
        - ww:     [B, L_pred, 3]，每个时间步对 3 个原型的权重（通常来自门控/混合系数）
        - q_hist: [B, d]，历史表示（例如 encoder 的 summary/query）

        写入流程：
        1) 在时间维上对 ww 求均值，得到样本级原型权重 p = mean_t ww ∈ [B,3]；
        2) 使用固定投影 P 将 q_hist 从 d 维降到 r 维：z_hist = normalize(q_hist @ P) ∈ [B,r]；
        3) 对每个原型 r_idx：
           - 按 p[:, r_idx] 对 z_hist 加权聚合得到该原型的“目标方向” z_r；
           - 用 EMA 更新原型：proto ← normalize((1-m)*proto + m*z_r)；
           - 累计 seen[r_idx]；
        4) 写入结束后执行一次轻量正交化 _orthogonalize_()。
        """
        p = ww.mean(dim=1)                      # [B, 3]
        z_hist = (q_hist @ self.P).contiguous() # [B, r]
        z_hist = F.normalize(z_hist, dim=-1)

        eps = 1e-6
        for r_idx in range(3):
            pr = p[:, r_idx:r_idx + 1]          # [B, 1]
            weight_sum = pr.sum()
            if float(weight_sum) > 0.0:
                z_r = (pr * z_hist).sum(dim=0) / (weight_sum + eps)  # [r]
                new_proto = F.normalize(
                    (1.0 - self.momentum) * self.z_protos[r_idx] + self.momentum * z_r,
                    dim=-1
                )
                self.z_protos[r_idx].copy_(new_proto)
                self.seen[r_idx] += weight_sum

        self._orthogonalize_()

    @torch.no_grad()
    def read_mixture(self, ww: torch.Tensor) -> torch.Tensor:
        """
        读取记忆（按步混合）：根据 ww 生成每个预测步的上下文向量，并升回 d 维。

        输入：
        - ww: [B, L_pred, 3]，每步对 3 原型的混合权重

        输出：
        - ctx_d: [B, L_pred, d]，按步上下文（先在 r 维混合，再用 P^T 升维）

        计算：
        - ctx_r = ww @ z_protos      -> [B, L, r]
        - ctx_d = ctx_r @ P.T        -> [B, L, d]
        """
        ctx_r = ww @ self.z_protos   # [B, L, r]
        ctx_d = ctx_r @ self.P.T     # [B, L, d]
        return ctx_d

    @torch.no_grad()
    def read_per_proto(self) -> torch.Tensor:
        """
        读取记忆（按原型）：返回 3 个原型各自升到 d 维后的向量。

        输出：
        - [3, d]，每个原型对应一个 d 维向量（z_protos @ P.T）
        """
        ctx_d = self.z_protos @ self.P.T  # [3, d]
        return ctx_d

    @torch.no_grad()
    def confidence_gate(self, ww: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        """
        基于熵的置信度门控：从 ww 的不确定性（熵）生成每步 gate 系数。

        直觉：
        - ww 越“尖”（越确定）熵越低，gate 越接近 1；
        - ww 越“平”（越不确定）熵越高，gate 越接近 0；
        - alpha 控制门控曲线的陡峭程度。

        输入：
        - ww: [B, L, 3]
        - alpha: 指数超参，默认 2.0

        输出：
        - gate: [B, L, 1]，公式：gate = (1 - H/ln3)^alpha
        """
        e = -(ww.clamp_min(1e-8) * ww.clamp_min(1e-8).log()).sum(dim=-1)  # [B, L]
        e = e / math.log(3.0)
        gate = (1.0 - e).pow(alpha).unsqueeze(-1)  # [B, L, 1]
        return gate

    @torch.no_grad()
    def reset(self):
        """
        重置记忆：将原型 z_protos 与计数器 seen 恢复到初始状态。

        操作：
        - z_protos 重新以 N(0, 1e-3) 初始化并做单位归一化；
        - seen 清零。
        """
        self.z_protos.normal_(std=1e-3)
        self.z_protos.copy_(F.normalize(self.z_protos, dim=-1))
        self.seen.zero_()




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
        self.use_decoding = config.use_decoding
        self.r = config.r
        self.c_in = c_in
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
            # Intra：共享的一套块（不再按 patch 复制参数）
            intra = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.2)
                for _ in range(self.num_layers_intra_patch)
            ])
            # Inter：保持不变
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

        # ---------- 记忆库（超轻瓶颈版） ----------
        if self.use_memory and self.mem_mode == 'tbm':
            self.memory = TinyBottleneckMemory(d_model=self.d_model, r=self.r, momentum=self.momentum)
            # 每个专家一个“幅度标量”（限幅 0.25*tanh）
            self.mem_stepA_raw = nn.Parameter(torch.tensor(0.0))
            self.mem_stepB_raw = nn.Parameter(torch.tensor(0.0))
            self.mem_stepC_raw = nn.Parameter(torch.tensor(0.0))
            # 记忆注入的随机失活，进一步防过拟合
            self.mem_dropout = nn.Dropout(p=0.2)
            # expert->proto 映射（可学习）：把 ww(专家权重) 映射到 proto 权重
            self.expert2proto = nn.Parameter(torch.eye(3))  # [3,3]，初始化为近似恒等映射

            # proto 权重的温度（可选，但很有用）
            self.proto_tau_raw = nn.Parameter(torch.tensor(0.0))  # tau = softplus + 1e-3

        # ---------- 三个专家头 ----------
        self.headA = nn.Linear(d_model, 1)
        self.headB = nn.Linear(d_model, 1)
        self.headC = nn.Linear(d_model, 1)
        
        self.enc_linear = nn.Linear(d_model, self.c_in)
        
        # Forecaster
        self.forecaster = ImplicitForecaster(self.config)

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
        intra_blocks: 共享 Intra（ModuleList，长度 = num_layers_intra_patch）
        """
        B = x_emb.size(0)

        # 1) 切 patch: [B, total_len, d] -> [B, P, pl, d]
        patches = rearrange(
            x_emb, 'b (p pl) d -> b p pl d',
            p=self.num_patches, pl=self.patch_len
        )  # [B, P, pl, d]

        # 2) Intra（共享参数）：合并 (B,P) -> (B*P)
        patches_intra = rearrange(patches, 'b p pl d -> (b p) pl d').contiguous()  # [B*P, pl, d]
        for block in intra_blocks:
            patches_intra = block(patches_intra, attn_mask=intra_mask)             # [B*P, pl, d]
        patches_intra = rearrange(patches_intra, '(b p) pl d -> b p pl d', b=B, p=self.num_patches).contiguous()

        # 展平回 token 序列
        intra_tokens = rearrange(patches_intra, 'b p pl d -> b (p pl) d')          # [B, total_len, d]

        # 3) Inter（保持你原逻辑）：在 patch 维做注意力
        inter_patches = rearrange(patches_intra, 'b p pl d -> (b pl) p d')         # [B*pl, P, d]
        for block in inter_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_mask)             # [B*pl, P, d]
        inter_tokens = rearrange(inter_patches, '(b pl) p d -> b (p pl) d', b=B, pl=self.patch_len)  # [B, total_len, d]

        # 4) 融合 + norm
        final = post_norm(intra_tokens + inter_tokens)
        return final


    def forward(self, x, x_mark=None, y_true=None, sample_ids=None):
        # x: [B, seq_len, d]  d = 8 
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

        # # 用 ww 做一个样本级的路由系数（平均到时间维）
        # mix = ww.mean(dim=1)           # [B, 3]
        # g0 = mix[:, 0:1].unsqueeze(-1) # [B, 1, 1]
        # g1 = mix[:, 1:2].unsqueeze(-1)
        # g2 = mix[:, 2:3].unsqueeze(-1)

        # # 这里用 (1 + g) 而不是直接乘 g，避免某一路完全被压成 0
        # x_embA = x_emb * (1.0 + g0)
        # x_embB = x_emb * (1.0 + g1)
        # x_embC = x_emb * (1.0 + g2)
        
        # ---------- 三个专家主干 ---------- [B, total_len, d_model] ----------
        finalA = self._forward_backbone(x_emb, self.intraA, self.interA, intra_mask, inter_mask, self.post_normA)  # [B, total_len, d]
        finalB = self._forward_backbone(x_emb, self.intraB, self.interB, intra_mask, inter_mask, self.post_normB)
        finalC = self._forward_backbone(x_emb, self.intraC, self.interC, intra_mask, inter_mask, self.post_normC)
        
        if self.use_memory and self.mem_mode == 'tbm':
            gate_conf = self.memory.confidence_gate(ww.detach())  # [B, pred_len, 1]

            # (1) 学习 expert->proto 映射：ww -> ww_proto
            # ww: [B,L,3]，expert2proto: [3,3] => logits: [B,L,3]
            proto_logits = ww.detach() @ self.expert2proto

            proto_tau = F.softplus(self.proto_tau_raw) + 1e-3
            ww_proto = torch.softmax(proto_logits / proto_tau, dim=-1)  # [B,L,3]

            # (2) 用 proto 权重做 read_mixture，得到按步上下文（更强）
            ctx = self.memory.read_mixture(ww_proto).detach()  # [B, L, d_model]
           

            base = self.mem_dropout(torch.tanh(ctx))  # [B,L,d_model]

            # (3) 仍然保留“专家逐步权重”注入到各自的尾段
            w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]
            sA = 0.6 * torch.tanh(self.mem_stepA_raw)
            sB = 0.6 * torch.tanh(self.mem_stepB_raw)
            sC = 0.6 * torch.tanh(self.mem_stepC_raw)

            injA = sA * (base * w0 * gate_conf)  # [B,L,d_model]
            injB = sB * (base * w1 * gate_conf)
            injC = sC * (base * w2 * gate_conf)

            def inject_tail(base_tokens, inj_tail):
                # base_tokens: [B,total_len,d_model], inj_tail: [B,pred_len,d_model]
                head = base_tokens[:, :-inj_tail.size(1), :]
                tail = base_tokens[:, -inj_tail.size(1):, :] + inj_tail
                return torch.cat([head, tail], dim=1)

            finalA = inject_tail(finalA, injA)
            finalB = inject_tail(finalB, injB)
            finalC = inject_tail(finalC, injC)

            # (4) 写入：用 proto 权重来更新原型（而不是直接用 ww），并用置信度抑制噪声写入
            if self.training:
                # q_hist = x_emb_hist.mean(dim=1).detach()        # [B,d_model]（也可换成 x_emb_hist[:,-1]）
                q_hist = (0.5 * x_emb_hist[:, -1, :] + 0.5 * x_emb_hist.mean(dim=1)).detach()

                # 置信度权重（样本级）: [B,1,1]
                conf = gate_conf.mean(dim=1, keepdim=True)      # [B,1,1]
                ww_write = ww_proto * conf                      # [B,L,3]
                self.memory.write(ww_write, q_hist)

                
        if self.use_decoding :
            finalA = self.enc_linear(finalA)
            finalB = self.enc_linear(finalB)
            finalC = self.enc_linear(finalC)
            
            finalA = finalA.permute(0,2,1)  # [B, c_in, total_len]
            finalB = finalB.permute(0,2,1)
            finalC = finalC.permute(0,2,1)

        if self.use_decoding :
            # ---------- 三个专家头，切出预测区间 ----------
            yA = self.forecaster(finalA, x)[:, :self.pred_len, :]  # [B, L, 1]
            yB = self.forecaster(finalB, x)[:, :self.pred_len, :]  # [B, L, 1]
            yC = self.forecaster(finalC, x)[:, :self.pred_len, :]  # [B, L, 1]
        else :
            # # ---------- 三个专家头，切出预测区间 ----------
            yA = self.headA(finalA)[:, -self.pred_len:, :]  # [B, L, 1]
            yB = self.headB(finalB)[:, -self.pred_len:, :]  # [B, L, 1]
            yC = self.headC(finalC)[:, -self.pred_len:, :]  # [B, L, 1]
           
        # ---------- ww 逐步加权融合 ----------
        w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]  # [B, L, 1] x 3
        y = w0 * yA + w1 * yB + w2 * yC                       # [B, L, 1]
        return y