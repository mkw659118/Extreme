#Author  :   mkw 
#Time    :   2025/09/17 17:27:29
#Desc    :   None

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Bucketed Extreme Memory (per patch position)
# =========================
class BucketedExtremeMemory(nn.Module):
    """
    每个 patch 位置一个“桶”（bucket）：
    - keys[p]: [M, d]   表示第 p 个桶的 M 个 key 向量
    - vals[p]: [M, d]   表示第 p 个桶的 M 个 value 向量
    作用：存储 patch 的极端记忆，供后续相似性检索和读写更新。
    """
    def __init__(self, num_buckets: int, d_model: int, mem_size_per_bucket: int = 256,
                 topk: int = 16, temperature: float = 0.5, ema_momentum: float = 0.2, device=None):
        super().__init__()
        self.P = num_buckets                  # 桶的数量（通常等于 patch 数）
        self.d = d_model                      # 每个向量的维度
        self.M = mem_size_per_bucket          # 每个桶里最多存储多少个向量
        self.K = topk                         # 读取时默认选取 top-k
        self.tau = temperature                # 温度系数，控制 softmax 的平滑度
        self.momentum = ema_momentum          # EMA 衰减系数，用于写入更新
        self.device = device                  # 运行设备

        # 初始化 keys，每个 bucket 一个 [M,d] 的参数矩阵，正态分布并 L2 归一化
        self.keys = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])

        # 初始化 vals，同样是每个 bucket 一个 [M,d] 的参数矩阵
        self.vals = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])

        # 环形写指针，每个 bucket 一个，记录写入位置
        self.register_buffer('ptr', torch.zeros(self.P, dtype=torch.long))

        # 标志：只用 EMA 更新（True 时不采用 FIFO 覆盖写）
        self.ema_only = True

    # ============= 写操作 =============
    @torch.no_grad()
    def write(self, p: int, k_batch: torch.Tensor, v_batch: torch.Tensor, w_batch: torch.Tensor = None):
        """
        写入 bucket p
        参数：
        - k_batch: [B,d] 批量的 key 向量
        - v_batch: [B,d] 批量的 value 向量
        - w_batch: [B] 或 [B,1] 权重，可选
        """
        if k_batch.numel() == 0:             # 空输入则直接返回
            return
        k_batch = F.normalize(k_batch, dim=-1)  # L2 归一化 key
        v_batch = F.normalize(v_batch, dim=-1)  # L2 归一化 value

        if self.ema_only:                    # ============ EMA 更新模式 ============
            if w_batch is None:              # 如果没有提供权重
                direction = k_batch.mean(dim=0, keepdim=True)   # 平均 key 向量
                value_dir = v_batch.mean(dim=0, keepdim=True)   # 平均 value 向量
            else:                            # 如果提供了权重
                ws = w_batch
                if ws.dim() == 1: ws = ws.unsqueeze(-1)         # 确保 [B,1] 形状
                ws = ws / (ws.sum(dim=0, keepdim=True) + 1e-8)  # 归一化权重
                direction = (ws.t() @ k_batch).mean(dim=0, keepdim=True)  # 加权 key
                value_dir = (ws.t() @ v_batch).mean(dim=0, keepdim=True)  # 加权 value

            # EMA 更新 keys 和 vals（凸组合 + 归一化）
            new_k = F.normalize((1 - self.momentum) * self.keys[p].data + self.momentum * direction, dim=-1)
            new_v = F.normalize((1 - self.momentum) * self.vals[p].data + self.momentum * value_dir, dim=-1)
            self.keys[p].data = new_k
            self.vals[p].data = new_v

        else:                                # ============ FIFO 覆盖写模式 ============
            b = k_batch.size(0)              # 批大小
            start = int(self.ptr[p].item())  # 当前写指针
            idx = (torch.arange(b, device=k_batch.device) + start) % self.M  # 环形索引
            self.keys[p].data[idx] = k_batch # 覆盖写入 keys
            self.vals[p].data[idx] = v_batch # 覆盖写入 vals
            self.ptr[p] = (self.ptr[p] + b) % self.M  # 更新写指针

    # ============= 读操作 =============
    def read(self, p: int, q: torch.Tensor, topk=None):
        """
        从 bucket p 中读取与 query q 最相关的向量
        参数：
        - q: [B,d] 查询向量
        返回：
        - m: [B,d] 聚合后的 value 表示
        - s: [B,1] 最大相似度分数
        - wK: [B,M] 每个 memory slot 的权重分布（稀疏）
        """
        if topk is None:
            topk = self.K

        Kmat = F.normalize(self.keys[p], dim=-1)   # [M,d] 当前 bucket 的 keys
        qn = F.normalize(q, dim=-1)                # [B,d] 归一化 query
        sim = (qn @ Kmat.t()) / self.tau           # [B,M] 计算相似度（缩放）

        k = min(topk, self.M)                      # 选取 top-k（不能超过 M）
        topv, topi = torch.topk(sim, k=k, dim=-1)  # [B,k] 相似度值和索引
        w = torch.softmax(topv, dim=-1)            # [B,k] 归一化权重

        Vmat = F.normalize(self.vals[p], dim=-1)   # [M,d] 当前 bucket 的 values
        picked = Vmat[topi]                        # [B,k,d] 取出对应的 values
        m = (w.unsqueeze(-1) * picked).sum(dim=1)  # [B,d] 加权聚合
        s, _ = sim.max(dim=-1, keepdim=True)       # [B,1] 最大相似度作为匹配强度

        wK = torch.zeros_like(sim)                 # [B,M] 初始化稀疏分布
        wK.scatter_(-1, topi, w)                   # 将 top-k 权重填入对应位置

        return m, s, wK
