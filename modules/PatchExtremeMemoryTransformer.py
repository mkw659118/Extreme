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



# 定义一个超轻瓶颈记忆模块类，继承自 nn.Module
class TinyBottleneckMemory(nn.Module):
    # 类的整体说明文档，描述该记忆模块的设计思想和功能
    """
    超轻瓶颈记忆：
    - 仅在 r 维（远小于 d_model）维护 3 个原型；
    - 上/下投影用固定正交矩阵（buffer，不训练）；
    - 读：按步混合 -> [B, L_pred, d]；
    - 写：EMA（no_grad），带轻量正交化；
    - 另提供按步置信度门控（基于 ww 熵）。
    """
    # 初始化函数，指定特征维度 d_model、瓶颈维 r、动量系数 momentum 和初始化标准差 init_std
    def __init__(self, d_model: int, r: int = 8, momentum: float = 0.05, init_std: float = 1e-3):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存输入特征维度 d_model
        self.d_model = d_model
        # 保存瓶颈维度 r
        self.r = r
        # 保存 EMA 更新的动量系数
        self.momentum = momentum

        # 在不计算梯度的环境中构造固定的正交投影矩阵 P
        with torch.no_grad():
            # 随机初始化一个形状为 [d_model, r] 的矩阵 A
            A = torch.randn(d_model, r)
            # 对 A 做 QR 分解，得到具有正交列的 Q（仅保留 reduced 形式）
            Q, _ = torch.linalg.qr(A, mode='reduced')  # Q: [d, r]
        # 将正交矩阵 Q 注册为 buffer，命名为 P，不参与训练更新
        self.register_buffer("P", Q)  # 不参与训练

        # 初始化瓶颈原型向量，共 3 个，每个为 r 维
        z = torch.randn(3, r) * init_std
        # 对原型向量在最后一维做 L2 归一化
        z = F.normalize(z, dim=-1)
        # 将原型向量注册为 buffer，命名为 z_protos
        self.register_buffer("z_protos", z)
        # 注册一个记录每个原型被“使用/更新”权重总和的计数器向量 seen
        self.register_buffer("seen", torch.zeros(3))

    # 使用装饰器表示该函数在调用时不计算梯度
    @torch.no_grad()
    # 定义内部正交化函数，对 3 个原型在 r 维空间做轻量 Gram-Schmidt 正交化
    def _orthogonalize_(self):
        # 从 z_protos 拷贝一份张量，避免在原 tensor 上直接做中间操作
        Z = self.z_protos.detach().clone()  # [3, r]
        # 遍历每一个原型向量索引 i
        for i in range(Z.size(0)):
            # 对第 i 个原型，依次减去其在之前所有原型（0..i-1）方向上的投影
            for j in range(i):
                # 计算第 i 个原型与第 j 个原型的内积系数 coef
                coef = torch.dot(Z[i], Z[j])
                # 从第 i 个原型中减去在第 j 个原型方向上的分量，实现正交
                Z[i] = Z[i] - coef * Z[j]
            # 对第 i 个原型做归一化，保持为单位向量
            Z[i] = F.normalize(Z[i], dim=-1)
        # 将正交化后的原型矩阵写回到 z_protos 中
        self.z_protos.copy_(Z)

    # 使用装饰器表示写入记忆操作不参与梯度计算
    @torch.no_grad()
    # 定义写操作函数，根据 ww 和 q_hist 更新瓶颈原型
    def write(self, ww: torch.Tensor, q_hist: torch.Tensor):
        # 函数文档：说明 ww 与 q_hist 的形状及整体更新逻辑
        """
        ww:     [B, L_pred, 3]
        q_hist: [B, d]
        把 q_hist 降到 r 维，再按 p=mean_t ww 聚合更新到 z_protos。
        """
        # 在时间维 L_pred 上做平均，得到每个样本对 3 个原型的平均权重 p
        p = ww.mean(dim=1)                             # [B, 3]
        # 使用投影矩阵 P 将历史 query 表示 q_hist 从 d 维降到 r 维
        z_hist = (q_hist @ self.P).contiguous()        # [B, r]
        # 对降维后的历史表示 z_hist 做归一化，突出方向信息
        z_hist = F.normalize(z_hist, dim=-1)

        # 定义一个很小的常数 eps，用于避免除零
        eps = 1e-6
        # 遍历每一个原型索引 r_idx ∈ {0,1,2}
        for r_idx in range(3):
            # 取出当前原型在 p 中对应的一列权重，形状 [B, 1]
            pr = p[:, r_idx:r_idx+1]                  # [B,1]
            # 计算该原型在当前 batch 中的总权重和
            weight_sum = pr.sum()
            # 只有当总权重大于 0 时才对该原型进行更新
            if float(weight_sum) > 0.0:
                # 使用权重 pr 对 z_hist 做加权平均，得到该原型在本批次的目标方向 z_r
                z_r = (pr * z_hist).sum(dim=0) / (weight_sum + eps)  # [r]
                # 使用 EMA 方式更新原型：旧原型与新方向按 momentum 插值后再归一化
                new_proto = F.normalize(
                    (1.0 - self.momentum) * self.z_protos[r_idx] + self.momentum * z_r, dim=-1
                )
                # 将更新后的原型向量写回 z_protos 中对应位置
                self.z_protos[r_idx].copy_(new_proto)
                # 累计该原型在本批次中被使用的权重和到 seen 计数器
                self.seen[r_idx] += weight_sum
        # 在每次写入之后，对全部原型执行一次轻量正交化，减少冗余
        self._orthogonalize_()

    @torch.no_grad()
    # 定义读操作函数，根据 ww 生成按步上下文并升回 d 维
    def read_mixture(self, ww: torch.Tensor) -> torch.Tensor:
        # 函数文档：说明输出为按时间步的上下文表示，形状 [B, L_pred, d]
        """
        返回按步上下文（升回 d 维）：[B, L_pred, d]
        """
        # 首先在 r 维瓶颈空间中混合原型： [B, L, 3] @ [3, r] -> [B, L, r]
        ctx_r = ww @ self.z_protos
        # 再通过投影矩阵的转置升维回 d 维： [B, L, r] @ [r, d] -> [B, L, d]
        ctx_d = ctx_r @ self.P.T
        # 返回升维后的上下文表示
        return ctx_d
    

    @torch.no_grad()
    def read_per_proto(self) -> torch.Tensor:
        """
        返回 3 个原型各自升到 d 维的向量: [3, d]
        """
        ctx_d = self.z_protos @ self.P.T   # [3, r] @ [r, d] -> [3, d]
        return ctx_d


    # 使用装饰器表示置信度门控计算不参与梯度
    @torch.no_grad()
    # 定义基于熵的置信度门控函数，为每个时间步生成一个 gate 系数
    def confidence_gate(self, ww: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # 函数文档：说明 gate 的计算方式与输出形状
        """
        基于熵的置信度门控：gate = (1 - H/ln3)^alpha，形状 [B, L, 1]
        """
        # 先对 ww 做 clamp，避免出现 log(0)，再按类别维计算熵：e = -sum(p log p)
        e = -(ww.clamp_min(1e-8) * ww.clamp_min(1e-8).log()).sum(dim=-1)  # [B, L]
        # 将熵 e 归一化到 [0,1]，除以 ln(3)（3 为类别数）
        e = e / math.log(3.0)
        # 根据公式 gate = (1 - e)^alpha 计算置信度，并在最后一维上扩展形状为 [B, L, 1]
        gate = (1.0 - e).pow(alpha).unsqueeze(-1)  # [B, L, 1]
        # 返回置信度门控系数 gate
        return gate

    # 使用装饰器表示重置操作不参与梯度计算
    @torch.no_grad()
    # 定义重置函数，将原型和 seen 计数器恢复到初始状态
    def reset(self):
        # 使用正态分布随机重新初始化 z_protos，标准差为 1e-3
        self.z_protos.normal_(std=1e-3)
        # 对重新初始化后的原型做归一化，保证每个原型为单位向量
        self.z_protos.copy_(F.normalize(self.z_protos, dim=-1))
        # 将 seen 计数器清零
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
            # Intra：每个 patch 一组堆叠的块
            intra = nn.ModuleList([
                nn.ModuleList([
                    TransformerBlock(d_model, num_heads, d_ff=None, dropout=0.2)
                    for _ in range(self.num_layers_intra_patch)
                ])
                for _ in range(self.num_patches)
            ])
            # Inter：跨 patch 的块
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
        返回:
            final: [B, total_len, d]
        这里实现的是：先对每个 patch 做 Intra，再在 patch 之间做 Inter（串联设计）
        """
        B = x_emb.size(0)

        # 1) 划分 patch
        patches = rearrange(
            x_emb, 'b (p pl) d -> b p pl d',
            p=self.num_patches, pl=self.patch_len
        )  # [B, P, pl, d]

        # 2) Intra: 对每个 patch 走自己的一组块，但不 in-place 写回原 view
        updated_patches = []
        for p in range(self.num_patches):
            out = patches[:, p, :, :]  # [B, pl, d]
            for block in intra_blocks[p]:
                out = block(out, attn_mask=intra_mask)  # [B, pl, d]
            updated_patches.append(out.unsqueeze(1))    # [B, 1, pl, d]

        # 拼成新的 patch 张量（已经是 Intra 之后的结果）
        patches_intra = torch.cat(updated_patches, dim=1)  # [B, P, pl, d]

        # 展平成 token 序列，作为 Intra 分支输出
        intra_tokens = rearrange(patches_intra, 'b p pl d -> b (p pl) d')  # [B, total_len, d]

        # 3) Inter: 在 patch 维做自注意力（用 Intra 之后的 patches_intra）
        inter_patches = rearrange(patches_intra, 'B P pl d -> (B pl) P d')  # [B*pl, P, d]
        for block in inter_blocks:
            inter_patches = block(inter_patches, attn_mask=inter_mask)      # [B*pl, P, d]
        inter_tokens = rearrange(
            inter_patches, '(B pl) P d -> B (P pl) d',
            B=B, pl=self.patch_len
        )  # [B, total_len, d]

        # 4) Intra + Inter 融合，再做层归一化
        final = intra_tokens + inter_tokens         # [B, total_len, d]
        final = post_norm(final)
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

        # 用 ww 做一个样本级的路由系数（平均到时间维）
        mix = ww.mean(dim=1)           # [B, 3]
        g0 = mix[:, 0:1].unsqueeze(-1) # [B, 1, 1]
        g1 = mix[:, 1:2].unsqueeze(-1)
        g2 = mix[:, 2:3].unsqueeze(-1)

        # 这里用 (1 + g) 而不是直接乘 g，避免某一路完全被压成 0
        x_embA = x_emb * (1.0 + g0)
        x_embB = x_emb * (1.0 + g1)
        x_embC = x_emb * (1.0 + g2)
        
        # ---------- 三个专家主干 ---------- [B, total_len, d_model] ----------
        finalA = self._forward_backbone(x_embA, self.intraA, self.interA, intra_mask, inter_mask, self.post_normA)  # [B, total_len, d]
        finalB = self._forward_backbone(x_embB, self.intraB, self.interB, intra_mask, inter_mask, self.post_normB)
        finalC = self._forward_backbone(x_embC, self.intraC, self.interC, intra_mask, inter_mask, self.post_normC)
        
        finalA = self.enc_linear(finalA)
        finalB = self.enc_linear(finalB)
        finalC = self.enc_linear(finalC)
        
        finalA = finalA.permute(0,2,1)  # [B, c_in, total_len]
        finalB = finalB.permute(0,2,1)
        finalC = finalC.permute(0,2,1)

        if self.use_memory and self.mem_mode == 'tbm':
            # 1) 基于熵的置信度门控（不反传）
            gate_conf = self.memory.confidence_gate(ww.detach())     # [B, pred_len, 1]

            # 2) 每个原型 ↔ 一个专家：读出 3 个原型的上下文，并展开到 [B, pred_len, d]
            ctx_protos = self.memory.read_per_proto().detach()       # [3, d]
            ctxA = ctx_protos[0].view(1, 1, -1).expand(B, self.pred_len, -1)  # [B, pred_len, d]
            ctxB = ctx_protos[1].view(1, 1, -1).expand(B, self.pred_len, -1)
            ctxC = ctx_protos[2].view(1, 1, -1).expand(B, self.pred_len, -1)

            # 3) 每个专家自己的“温和”残差（tanh 限幅 + dropout）
            baseA = self.mem_dropout(torch.tanh(ctxA))
            baseB = self.mem_dropout(torch.tanh(ctxB))
            baseC = self.mem_dropout(torch.tanh(ctxC))

            # 4) 专家通道权重（逐步）与幅度标量（0.25*tanh）
            w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]     # [B, pred_len, 1]
            sA = 0.25 * torch.tanh(self.mem_stepA_raw)                # 标量
            sB = 0.25 * torch.tanh(self.mem_stepB_raw)
            sC = 0.25 * torch.tanh(self.mem_stepC_raw)

            injA = sA * (baseA * w0 * gate_conf)                      # [B, pred_len, d]
            injB = sB * (baseB * w1 * gate_conf)
            injC = sC * (baseC * w2 * gate_conf)

            # 5) 只替换预测尾段
            def inject_tail_nopad(base, inj):
                B_, L_, D_ = base.shape
                Linj = inj.size(1)
                if Linj == 0:
                    return base
                head = base[:, :L_ - Linj, :]
                tail = base[:, L_ - Linj:, :] + inj
                return torch.cat([head, tail], dim=1)

            finalA = inject_tail_nopad(finalA, injA)
            finalB = inject_tail_nopad(finalB, injB)
            finalC = inject_tail_nopad(finalC, injC)

            # 6) 写回：仍然用历史均值（不反传）
            if self.training:
                q_hist = x_emb_hist.mean(dim=1).detach()
                self.memory.write(ww.detach(), q_hist)


        # # ---------- 三个专家头，切出预测区间 ----------
        # yA = self.headA(finalA)[:, -self.pred_len:, :]  # [B, L, 1]
        # yB = self.headB(finalB)[:, -self.pred_len:, :]  # [B, L, 1]
        # yC = self.headC(finalC)[:, -self.pred_len:, :]  # [B, L, 1]
        # ---------- 三个专家头，切出预测区间 ----------
        yA = self.forecaster(finalA, x)[:, :self.pred_len, :]  # [B, L, 1]
        yB = self.forecaster(finalB, x)[:, :self.pred_len, :]  # [B, L, 1]
        yC = self.forecaster(finalC, x)[:, :self.pred_len, :]  # [B, L, 1]

        # ---------- ww 逐步加权融合 ----------
        w0, w1, w2 = ww[..., 0:1], ww[..., 1:2], ww[..., 2:3]  # [B, L, 1] x 3
        y = w0 * yA + w1 * yB + w2 * yC                       # [B, L, 1]
        
        return y