# coding : utf-8
# Author : yuxiang Zeng
import math
import torch

from layers.att.external_attention import ExternalAttention
from layers.att.groupquery_attention import GroupQueryAttention
from layers.att.multilatent_attention import MLA
from layers.att.multiquery_attention import MultiQueryAttentionBatched
from layers.feedforward.ffn import FeedForward
from layers.feedforward.moe import MoE
from layers.feedforward.smoe import SparseMoE
from einops import rearrange

from layers.revin import RevIN


class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.10):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)

    def forward(self, x, attn_mask=None, weight=False):
        out, weights = self.att(x, x, x, attn_mask=attn_mask)
        return (out, weights) if weight else out


def get_norm(d_model, method):
    if method == 'batch':
        return torch.nn.BatchNorm1d(d_model)
    elif method == 'layer':
        return torch.nn.LayerNorm(d_model)
    elif method == 'rms':
        return torch.nn.RMSNorm(d_model)
    return None


def get_ffn(d_model, method):
    if method == 'ffn':
        return FeedForward(d_model, d_ff=d_model * 2, dropout=0.10)
    elif method == 'moe':
        return MoE(d_model=d_model, d_ff=d_model, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2, loss_coef=0.001)
    elif method == 'smoe':
        return SparseMoE(d_model=d_model, d_ff=d_model, num_experts=8, noisy_gating=True, num_k=2, loss_coef=0.001)
    return None


def get_att(d_model, num_heads, method):
    if method == 'self':
        return Attention(d_model, num_heads, dropout=0.10)
    elif method == 'external':
        return ExternalAttention(d_model, S=d_model*2)
    elif method == 'mla':
        return MLA(d_model, S=d_model*2)
    elif method == 'gqa':
        return GroupQueryAttention(d_model, S=d_model*2)
    elif method == 'mqa':
        return MultiQueryAttentionBatched(d_model, S=d_model*2)
    return None

class TokenEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = torch.nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
    
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FixedEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = torch.nn.Embedding(c_in, d_model)
        self.emb.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    

class TemporalEmbedding(torch.nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else torch.nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            # self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        #  + hour_x + minute_x
        return month_x + day_x + weekday_x + hour_x
    
    

class TimeFeatureEmbedding(torch.nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = torch.nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
    
class DataEmbedding(torch.nn.Module):
    def __init__(self, c_in, d_model, match_mode, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.match_mode = match_mode

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.value_embedding = torch.nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x_out = self.value_embedding(x) + self.position_embedding(x)
        else:
            # a 
            # a b
            # a c
            # a b c
            if self.match_mode == 'a':
                x_out = self.value_embedding(x)
            elif self.match_mode == 'ab':
                x_out = self.value_embedding(x) + self.position_embedding(x)
            elif self.match_mode == 'ac':
                x_out = self.value_embedding(x) + self.temporal_embedding(x_mark)
            elif self.match_mode == 'abc':
                x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
            else:
                raise ValueError(f"Unknown ablation mode: {self.match_mode}")

        return self.dropout(x_out)


# def generate_causal_window_mask(seq_len, win_size, device):
#     mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)  # 上三角屏蔽未来
#     for i in range(seq_len):
#         left = max(0, i - win_size + 1)
#         mask[i, :left] = True  # 屏蔽太早的历史
#     return mask.masked_fill(mask, float('-inf'))  #  保持为 (T, T)

# [REPLACED] —— 生成“加性 mask（允许=0, 屏蔽=-inf）”，兼容 MHA/SDPA
def generate_causal_window_mask(seq_len, win_size, device, dtype=torch.float32):
    """
    返回加性mask: 允许=0，屏蔽=-inf（上三角=未来；窗口外=太早历史）
    形状: [seq_len, seq_len]，可直接传入 nn.MultiheadAttention / SDPA
    """
    bad = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)  # 未来全屏蔽
    # 窗口外（过早的历史）屏蔽
    for i in range(seq_len):
        left = max(0, i - win_size + 1)
        bad[i, :left] = True

    attn_bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    attn_bias.masked_fill_(bad, torch.finfo(attn_bias.dtype).min)  # -inf
    return attn_bias

class Transformer(torch.nn.Module):
    def __init__(self, input_size, d_model, revin, num_heads, num_layers,
                 seq_len, pred_len, match_mode, win_size, patch_len, device,
                 norm_method='layer', ffn_method='ffn', att_method='self',):
        super().__init__()
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.match_mode = match_mode
        self.win_size = win_size
        self.device = device
        self.att_method = att_method
        self.total_len = seq_len + pred_len
        self.patch_len = patch_len
        assert self.total_len % patch_len == 0, "total_len must be divisible by patch_len"
        self.num_patches = self.total_len // patch_len

        if self.revin:
            self.revin_layer = RevIN(num_features=input_size, affine=False, subtract_last=False)

        self.enc_embedding = DataEmbedding(input_size, d_model, self.match_mode)

        # 1. 映射到 seq_len + pred_len 长度
        self.predict_linear = torch.nn.Linear(seq_len, self.total_len)

        # 2. 为每个 patch 分配一个独立的 Transformer block
        self.patch_transformers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    get_norm(d_model, norm_method),
                    get_att(d_model, num_heads, att_method),
                    get_norm(d_model, norm_method),
                    get_ffn(d_model, ffn_method)
                ) for _ in range(num_layers)
            ]) for _ in range(self.num_patches)
        ])


        # [REPLACED]：原先是 self.patch_inter_transformer = nn.Sequential(...)
        # 现在改成 ModuleList，方便 forward 时传入 inter_mask
        self.patch_inter_blocks = torch.nn.ModuleList([  # [REPLACED]
            torch.nn.Sequential(
                get_norm(d_model, norm_method),
                get_att(d_model, num_heads, att_method),
                get_norm(d_model, norm_method),
                get_ffn(d_model, ffn_method)
            ) for _ in range(num_layers)
        ])


        self.norm = get_norm(d_model, norm_method)
        
        self.projection = torch.nn.Linear(d_model, input_size)

    # [NEW] 一次性生成 patch 内/patch 间两类 mask
    def _build_masks(self, device, dtype=torch.float32):  # [NEW]
        # patch 内（步级别）mask
        patch_mask = generate_causal_window_mask(self.patch_len, self.win_size, device, dtype)
        # patch 间（以 patch 为粒度）窗口：把步级窗口折算到 patch 级别
        inter_win = max(1, math.ceil(self.win_size / self.patch_len))
        inter_mask = generate_causal_window_mask(self.num_patches, inter_win, device, dtype)
        return patch_mask, inter_mask


    def forward(self, x, x_mark=None):
        if self.revin:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        # [B, L, C] → [B, L, d_model]
        x = self.enc_embedding(x, x_mark)

        # [B, L, d_model] → [B, d_model, L] → project to total_len
        x = rearrange(x, 'b l d -> b d l')  # for predict_linear
        x = self.predict_linear(x)         # [B, d_model, total_len]
        x = rearrange(x, 'b d l -> b l d')  # back to [B, total_len, d_model]

        # Split into patches: [B, total_len, d_model] → [B, num_patches, patch_len, d_model]
        x = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)

        # [NEW] 生成两种掩码
        patch_mask, inter_mask = self._build_masks(x.device, x.dtype)  # [NEW]
        
        # Pass each patch into its own Transformer block
        outs = []
        for i in range(self.num_patches):
            patch = x[:, i]  # [B, patch_len, d_model]
            out = patch
            for block in self.patch_transformers[i]:
                norm1, attn, norm2, ff = block
                # [CHANGED] 仅当使用标准 MHA('self') 时传入 patch_mask
                if self.att_method == 'self':  # [CHANGED]
                    out = attn(norm1(out), attn_mask=patch_mask) + out
                else:
                    out = attn(norm1(out)) + out
                out = ff(norm2(out)) + out
            out = self.norm(out)  # optional
            outs.append(out)

        # Patch 内处理后拼接
        x = torch.cat(outs, dim=1)  # [B, total_len, d_model]

        # Patch 表征 pooling
        x_patch_tokens = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)
        x_patch_tokens = x_patch_tokens.mean(dim=2)  # [B, num_patches, d_model]

        # [NEW] 手动展开 patch 间注意力，传入 inter_mask
        out_tokens = x_patch_tokens
        for block in self.patch_inter_blocks:  # [NEW]
            norm1, attn, norm2, ff = block
            if self.att_method == 'self':
                out_tokens = attn(norm1(out_tokens), attn_mask=inter_mask) + out_tokens
            else:
                out_tokens = attn(norm1(out_tokens)) + out_tokens
            out_tokens = ff(norm2(out_tokens)) + out_tokens

        # 融合 patch 间上下文
        x_patch_ctx = out_tokens.unsqueeze(2).repeat(1, 1, self.patch_len, 1)  # [B, np, pl, d]
        x = rearrange(x, 'b (np pl) d -> b np pl d', np=self.num_patches, pl=self.patch_len)
        x = x + x_patch_ctx  # 加残差
        x = rearrange(x, 'b np pl d -> b (np pl) d')  # [B, total_len, d_model]

        # Project back to input_size
        y = self.projection(x)  # [B, total_len, input_size]

        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.total_len, 1)

        return y[:, -self.pred_len:, :]
