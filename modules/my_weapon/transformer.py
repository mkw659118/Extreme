# coding : utf-8
# Author : yuxiang Zeng
import torch
from einops import rearrange


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
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

class Transformer(torch.nn.Module):
    def __init__(self, dim, num_heads, num_layers, dropout = 0.):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, num_heads, dropout = dropout),
                FeedForward(dim, dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


if __name__ == '__main__':
    rank = 64
    num_layers = 4
    num_heads = 8
    dropout = 0.1
    model = Transformer(rank, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    x = torch.randn(16, 10, rank)
    output = model(x)
    print("Output shape:", output.shape)