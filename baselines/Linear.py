# coding : utf-8
# Author : Yuxiang Zeng
import torch
from einops import rearrange

from layers.revin import RevIN

class Linear(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # self.
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x, x_mark):
        # x: [B, L, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # rearrange to [B, D, L] to apply Linear on seq_len dimension
        x = rearrange(x, 'bs seq_len d_model -> bs d_model seq_len')
        y = self.predict_linear(x)  # [B, D, pred_len]
        y = rearrange(y, 'bs d_model pred_len -> bs pred_len d_model')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        # shape = [B, pred_len, D]
        return y