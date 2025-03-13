# coding : utf-8
# Author : yuxiang Zeng
import torch

from layers.att.external_attention import ExternalAttention
from layers.feedforward.ffn import FeedForward
from layers.feedforward.moe import MoE
from layers.att.self_attention import SelfAttention

def get_norm(d_model, method):
    if method == 'batch':
        return torch.nn.BatchNorm1d(d_model)
    elif method == 'layer':
        return torch.nn.LayerNorm(d_model)
    elif method == 'rms':
        return torch.nn.RMSNorm(d_model)

def get_ffn(d_model, method):
    if method == 'ffn':
        return FeedForward(d_model, d_ff=d_model * 2, dropout=0.10)
    elif method == 'moe':
        return MoE(d_model=d_model, d_ff=d_model * 2, d_out=d_model, num_shared_experts=2, num_routed_experts=4, topk=2, noise_std=0.1)

def get_att(d_model, num_heads, method):
    if method == 'self':
        return SelfAttention(d_model, num_heads, dropout=0.10)
    elif method == 'external':
        return ExternalAttention(d_model, S=d_model*2)

class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_layers, norm_method='rmsnorm', ffn_method='moe', att_method='self'):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        get_norm(d_model, norm_method),
                        get_att(d_model, num_heads, att_method),
                        get_norm(d_model, norm_method),
                        get_ffn(d_model, ffn_method)
                    ]
                )
            )
        self.norm = get_norm(d_model, norm_method)

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return self.norm(x)

if __name__ == '__main__':
    rank, num_layers, num_heads = 64, 4, 8
    model = Transformer(rank, num_heads=num_heads, num_layers=num_layers)
    x = torch.randn(16, 10, rank)
    output = model(x)
    print("Output shape:", output.shape)