# coding : utf-8
# Author : Yuxiang Zeng
import torch
from layers.revin import RevIN


class Linear2(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear2, self).__init__()
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.hidden_dim = config.hidden_dim

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(config.seq_len, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.LayerNorm(config.hidden_dim),
            torch.nn.Linear(config.hidden_dim, config.pred_len)
        )

    def forward(self, x, x_mark):
        # x: [B, L]
        if self.revin:
            x = x.unsqueeze(-1)
            x = self.revin_layer(x, 'norm')
            x = x.squeeze(-1)

        y = self.model(x)

        if self.revin:
            y = y.unsqueeze(-1)
            y = self.revin_layer(y, 'denorm')
            y = y.squeeze(-1)

        return y
