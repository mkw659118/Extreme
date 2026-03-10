# coding : utf-8
# Author : Yuxiang Zeng
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        ctx = torch.matmul(attn, v)  # [B,P,H]
        return ctx, attn