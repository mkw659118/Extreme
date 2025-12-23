from data_provider import DS
from utils import *
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd

from utils.activation import ALU

class RevIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.affine:
            self.affine_w = nn.Parameter(data=torch.ones(self.config.enc_in))
            self.affine_b = nn.Parameter(data=torch.zeros(self.config.enc_in))

    def forward(self, x, mode=None):
        if self.config.revin:
            if mode == 'norm':
                self._get_statistics(x)
                return self._normalize(x)
            elif mode == 'denorm':
                return self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim = tuple(range(1, x.ndim - 1))
        self.means = torch.mean(x, dim=dim, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim, correction=0, keepdim=True) + self.config.eps).detach()

    def _normalize(self, x):
        x = x - self.means
        x = x / self.stdev
        if self.config.affine:
            x = self.affine_w * x + self.affine_b
        return x

    def _denormalize(self, x):
        if self.config.affine:
            x = (x - self.affine_b) / (self.affine_w + self.config.eps * self.config.eps)
        x = x * self.stdev
        x = x + self.means
        return x


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(self.config.seq_len, self.config.d_model, bias=True)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, x_mark):
        embedding = self.dropout(self.embedding(x.permute(0, 2, 1)))
        return embedding


class FullAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, Q, K, V):
        B, L, H, E = Q.shape
        _, S, _, D = V.shape
        A = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(A / math.sqrt(E), dim=-1))
        x = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        return x


class AttnLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = FullAttn(self.config)
        d_heads = self.config.d_model // self.config.n_heads
        self.WQ = nn.Linear(self.config.d_model, self.config.n_heads * d_heads)
        self.WK = nn.Linear(self.config.d_model, self.config.n_heads * d_heads)
        self.WV = nn.Linear(self.config.d_model, self.config.n_heads * d_heads)
        self.WO = nn.Linear(self.config.n_heads * d_heads, self.config.d_model)

    def forward(self, Q, K, V):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        H = self.config.n_heads
        Q = self.WQ(Q).view(B, L, H, -1)
        K = self.WK(K).view(B, S, H, -1)
        V = self.WV(V).view(B, S, H, -1)
        x = self.attn(Q, K, V)
        x = self.WO(x.view(B, L, -1))
        return x


class FeedLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feed = nn.Sequential(
            nn.Conv1d(self.config.d_model, self.config.d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Conv1d(self.config.d_ff, self.config.d_model, kernel_size=1)
        )

    def forward(self, x):
        x = self.feed(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_layer = AttnLayer(self.config)
        self.feed_layer = FeedLayer(self.config)
        if self.config.network_norm == 'instance':
            self.norma = nn.InstanceNorm1d(self.config.enc_in, affine=True)
            self.normf = nn.InstanceNorm1d(self.config.enc_in, affine=True)
        elif self.config.network_norm == 'layer':
            self.norma = nn.LayerNorm(self.config.d_model)
            self.normf = nn.LayerNorm(self.config.d_model)
        else:
            self.norma = nn.Identity()
            self.normf = nn.Identity()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, Q, K, V, C):
        A = self.norma(C + self.dropout(self.attn_layer(Q, K, V)))
        x = self.normf(A + self.dropout(self.feed_layer(A)))
        return x, A


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_layer = TransformerLayer(self.config)

    def forward(self, x):
        x, _ = self.transformer_layer(x, x, x, x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.e_layers)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


def prepare_data(config):
    # data prepare
    trainX = pd.read_csv(
        "./datasets/" + config.dataset + "/" + config.reservoir_sensor + ".tsv", sep="\t"
    )
    trainX.columns = ["datetime", "value"]
    trainX.sort_values("datetime", inplace=True)
    return trainX

class AHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # in_features = self.config.d_model + self.config.seq_len // 2 + 1
        in_features = 432 + self.config.seq_len // 2 + 1
        out_features = self.config.spectrum_size // 2 + 1
        self.amplitude_head = nn.Sequential(
            nn.Linear(in_features, self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, out_features)
        )
        self.activation = ALU(w=0.5)
        # self._get_spectrum_prior()

    def _get_spectrum_prior(self):
        
        trainX = prepare_data()
        datamodule = DS(self.config, trainX)
        train_data = datamodule.get_diff_data()
        
        # train_data = TSFactory(self.config)('train')[0].x
        
        spectrum_prior = torch.zeros(1, self.config.enc_in, self.config.spectrum_size // 2 + 1)
        for i in range(len(train_data) - self.config.spectrum_size):
            x = train_data[i:i + self.config.spectrum_size].unsqueeze(0)
            if self.config.revin:
                dim = tuple(range(1, x.ndim - 1))
                means = torch.mean(x, dim=dim, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(x, dim=dim, correction=0, keepdim=True) + self.config.eps).detach()
                x = (x - means) / stdev
            spectrum_prior += torch.abs(torch.fft.rfft(x.permute(0, 2, 1), norm='ortho'))
        self.spectrum_prior = spectrum_prior / (len(train_data) - self.config.spectrum_size)

    def forward(self, x):
        # spectrum_prior = self.spectrum_prior.to(x.device).repeat(x.shape[0], 1, 1)
        amplitude = self.activation(self.amplitude_head(x))
        return amplitude


class PHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # in_features = self.config.d_model + self.config.seq_len // 2 + 1
        in_features = 432+ self.config.seq_len // 2 + 1
        out_features = self.config.spectrum_size // 2 + 1
        self.sin_head = nn.Sequential(
            nn.Linear(in_features, self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, out_features),
            nn.Tanh()
        )
        self.cos_head = nn.Sequential(
            nn.Linear(in_features, self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        sin = self.sin_head(x)
        cos = self.cos_head(x)
        phase = torch.atan2(sin, cos)
        return phase


class ImplicitForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.a_head = AHead(self.config)
        self.p_head = PHead(self.config)

    def forward(self, x_enc, x):
        fft_x = torch.fft.rfft(x.permute(0, 2, 1), norm='ortho')
        amp_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        amp_out = self.a_head(torch.cat((x_enc, amp_x), dim=-1))
        pha_out = self.p_head(torch.cat((x_enc, pha_x), dim=-1))
        x = torch.fft.irfft(amp_out * torch.exp(1j * pha_out), norm='ortho')
        return x.permute(0, 2, 1)

