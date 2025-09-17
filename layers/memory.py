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
    每个 patch 位置一个桶：
    - keys[p]: [M, d]
    - vals[p]: [M, d]
    """
    def __init__(self, num_buckets: int, d_model: int, mem_size_per_bucket: int = 256,
                 topk: int = 16, temperature: float = 0.5, ema_momentum: float = 0.2, device=None):
        super().__init__()
        self.P = num_buckets
        self.d = d_model
        self.M = mem_size_per_bucket
        self.K = topk
        self.tau = temperature
        self.momentum = ema_momentum
        self.device = device

        self.keys = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])
        self.vals = nn.ParameterList([
            nn.Parameter(F.normalize(torch.randn(self.M, self.d), dim=-1))
            for _ in range(self.P)
        ])
        self.register_buffer('ptr', torch.zeros(self.P, dtype=torch.long))
        self.ema_only = True

    @torch.no_grad()
    def write(self, p: int, k_batch: torch.Tensor, v_batch: torch.Tensor, w_batch: torch.Tensor = None):
        if k_batch.numel() == 0:
            return
        k_batch = F.normalize(k_batch, dim=-1)
        v_batch = F.normalize(v_batch, dim=-1)

        if self.ema_only:
            if w_batch is None:
                direction = k_batch.mean(dim=0, keepdim=True)   # [1,d]
                value_dir = v_batch.mean(dim=0, keepdim=True)   # [1,d]
            else:
                ws = w_batch
                if ws.dim() == 1: ws = ws.unsqueeze(-1)
                ws = ws / (ws.sum(dim=0, keepdim=True) + 1e-8)
                direction = (ws.t() @ k_batch).mean(dim=0, keepdim=True)
                value_dir = (ws.t() @ v_batch).mean(dim=0, keepdim=True)
            new_k = F.normalize((1 - self.momentum) * self.keys[p].data + self.momentum * direction, dim=-1)
            new_v = F.normalize((1 - self.momentum) * self.vals[p].data + self.momentum * value_dir, dim=-1)
            self.keys[p].data = new_k
            self.vals[p].data = new_v
        else:
            b = k_batch.size(0)
            start = int(self.ptr[p].item())
            idx = (torch.arange(b, device=k_batch.device) + start) % self.M
            self.keys[p].data[idx] = k_batch
            self.vals[p].data[idx] = v_batch
            self.ptr[p] = (self.ptr[p] + b) % self.M

    def read(self, p: int, q: torch.Tensor, topk=None):
        if topk is None:
            topk = self.K
        Kmat = F.normalize(self.keys[p], dim=-1)       # [M,d]
        qn = F.normalize(q, dim=-1)                    # [B,d]
        sim = (qn @ Kmat.t()) / self.tau               # [B,M]
        k = min(topk, self.M)
        topv, topi = torch.topk(sim, k=k, dim=-1)      # [B,k],[B,k]
        w = torch.softmax(topv, dim=-1)                # [B,k]
        Vmat = F.normalize(self.vals[p], dim=-1)       # [M,d]
        picked = Vmat[topi]                             # [B,k,d]
        m = (w.unsqueeze(-1) * picked).sum(dim=1)      # [B,d]
        s, _ = sim.max(dim=-1, keepdim=True)           # [B,1]

        wK = torch.zeros_like(sim)
        wK.scatter_(-1, topi, w)
        return m, s, wK
