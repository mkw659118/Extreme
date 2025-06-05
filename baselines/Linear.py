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
        self.d_model = config.d_model

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # self.
        self.middle_linear = torch.nn.Linear(config.seq_len, config.d_model)
        self.predict_linear = torch.nn.Linear(config.d_model, config.pred_len)


    def forward(self, x, x_mark):
        # x: [B, L, D]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # rearrange to [B, D, L] to apply Linear on seq_len dimension
        x = rearrange(x, 'B L D -> B D L')
        y = self.middle_linear(x)  # [B D d_model]
        y = self.predict_linear(y)  # [B, D, P]
        y = rearrange(y, 'B D P -> B P D')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        # shape = [B, P, D]
        return y