# coding: utf-8
# Author: mkw
# Date: 2025-06-06 11:58
# Description: Linear5（增加 Dropout 和 LayerNorm）

import torch
from einops import rearrange
from torch import nn

from layers.revin import RevIN


class Linear5(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear5, self).__init__()
        self.config = config
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.group = config.group
        self.input_size = config.input_size

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # 增加 Dropout 和激活函数
        self.linear_up = nn.Sequential(
            nn.Linear(self.seq_len // self.group, self.d_model),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )

        # 增加 LayerNorm
        self.linear_final = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, x_mark=None):
        # x: [B, L, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # L = g * l
        x = rearrange(x, 'B (g l) D -> B g l D', g=self.group)  # [B, g, l, D]
        x = rearrange(x, 'B g l D -> B g D l')  # [B, g, D, l]
        x = self.linear_up(x)  # [B, g, D, d_model]

        # [bs, patch, d_model, seq_len], torch.cat 
        x = x.sum(dim=1)  # [B, D, d_model]
        # [bs, seq_len, patch * d_model] linear-> [bs, seq_len, d_model]

        x = self.linear_final(x)  # [B, D, pred_len]

        x = rearrange(x, 'B L D -> B D L')  # [B, pred_len, D]

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
