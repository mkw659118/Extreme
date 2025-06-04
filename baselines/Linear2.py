# coding : utf-8
# Author : Yuxiang Zeng
import torch
from layers.revin import RevIN
from einops import rearrange, repeat

class Linear2(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear2, self).__init__()
        self.config = config
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.hidden_dim = config.hidden_dim

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(config.seq_len, config.hidden_dim),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm(config.hidden_dim),
            torch.nn.Linear(config.hidden_dim, config.pred_len)
        )

    def forward(self, x, x_mark):
        # x: [B, L, 1]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = rearrange(x, 'bs seq_len d_model -> bs d_model seq_len')
        y = self.model(x)
        y = rearrange(y, 'bs d_model pred_len -> bs pred_len d_model')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        # shape = [bs, pred_len, d=21]
        return y