import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from math import sqrt, log
import matplotlib.pyplot as plt
from utils.masking import TriangularCausalMask

from layers.Embed import DataEmbedding_wo_time

import random

def l2norm(t):
    return F.normalize(t, dim = -1)

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # m * \theta


    # freqs = [x, y]
    # freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis



def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) :
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)


    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)


    # xq_out.shape = [batch_size, seq_len, dim]
    freqs_cis = freqs_cis.to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,patch_n=65):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        number=(patch_n+1)//2
        self.freqs_cis = precompute_freqs_cis(d_model, number * 2)
    def forward(self, queries, keys, values, attn_mask, attn_bias,enc_rope=1):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        if enc_rope==1:
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis=self.freqs_cis)
        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            attn_bias
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, configs=None,
                 attn_scale_init=20):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.enc_in = configs.enc_in
       
        self.scale = scale

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            A = self.dropout(torch.softmax(scores * scale + attn_bias, dim=-1))
        else:
            A = self.dropout(torch.softmax(scores * scale, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, attn_bias=None,enc_rope=1):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            attn_bias=attn_bias,enc_rope=enc_rope
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = x + y
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, attn_bias=None,enc_rope=0):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask,enc_rope=enc_rope)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias,enc_rope=enc_rope)
                attns.append(attn)

        if self.norm is not None:
            # x = self.norm(x)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attns

class HMformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, device=0):
        super(HMformer, self).__init__()

        self.enc_in = configs.enc_in
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.patch_num += 1
        self.f=configs.fusion
        #self.padding_patch_layer_2 = nn.ReplicationPad1d((0, self.stride))
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = True if configs.ifatten == 1 else False
        self.num_heads = configs.n_heads
        self.factor = 3
        self.activation = 'gelu'

        self.d_model = configs.d_model
        self.padding_patch_layers=nn.ModuleList()
        self.enc_embeddings=nn.ModuleList()
        self.encoders=nn.ModuleList()
        self.convs=nn.ModuleList()
        self.projs=nn.ModuleList()
        for i in range(self.f):
            self.padding_patch_layers.append(nn.ReplicationPad1d((0, self.stride*(2**i))))
            self.enc_embeddings.append(DataEmbedding_wo_time(self.patch_size*(2**i),
                                            configs.d_model*(2**i), configs.embed, configs.freq, configs.dropout))
            self.encoders.append(Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention,
                                      configs=configs),
                                      configs.d_model*(2**i), configs.n_heads,patch_n=self.patch_num/(2**i)),
                    configs.d_model*(2**i),
                    configs.d_ff*(2**i),
                    dropout=configs.dropout,
                    activation=self.activation,

                ) for l in range(configs.e_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(configs.d_model)
            norm_layer=torch.nn.BatchNorm1d(configs.d_model*(2**i))
            ))
            self.projs.append(nn.Linear(configs.d_model * self.patch_num, configs.pred_len, bias=True))
            if i!=self.f-1:
                self.convs.append(nn.Conv1d(in_channels=configs.d_model*(2**i), out_channels=configs.d_model*(2**(i+1)), kernel_size=2, stride=2))







        self.cnt = 0
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x_enc /= stdev
        x_con_temp=[]
        x_out_temp=[]
        attnss=[]
        x_enc = rearrange(x_enc, 'b l m -> b m l')
        for i in range(self.f):
            x_enc_temp = self.padding_patch_layers[i](x_enc)
            x_enc_temp = x_enc_temp.unfold(dimension=-1, size=self.patch_size*(2**i), step=self.stride*(2**i))
            x_enc_temp = rearrange(x_enc_temp, 'b m n p -> (b m) n p')

            x_enc_temp = self.enc_embeddings[i](x_enc_temp).reshape(-1,x_enc_temp.shape[1],self.d_model*(2**i))
            if i!=0:
                x_enc_temp = x_enc_temp+x_con_temp[i-1]

            enc_out, attns = self.encoders[i]( x_enc_temp , attn_mask=None)
            attnss.append(attns)
            x_out_temp.append(self.projs[i](enc_out[:, :, :].reshape(B * M, -1)))
            if i!=self.f-1:
                x_con_temp.append(self.convs[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1))



        enc_out = torch.sum(torch.stack(x_out_temp, dim=0), dim=0)
        enc_out = rearrange(enc_out, '(b m) l -> b l m', m=M)

        # revin
        enc_out = enc_out[:, -self.pred_len:, :]
        enc_out = enc_out * stdev
        enc_out = enc_out + means


        if self.output_attention:
            return enc_out, attnss
        else:
            return enc_out  # [B, L, D]

