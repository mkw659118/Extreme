# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange


class SelfAttention(torch.nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = torch.nn.Softmax(dim=-1)
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

if __name__ == '__main__':
    inputs = torch.randn(1, 10, 50)
    model = SelfAttention(dim = 50, heads = 8, dim_head = 64)
    out = model(inputs)
    print(out.shape)