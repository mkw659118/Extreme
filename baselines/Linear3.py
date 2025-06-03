# coding : utf-8
# Author : Yuxiang Zeng
import torch
from layers.revin import RevIN


class Linear3(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Linear3, self).__init__()
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.hidden_dim = config.hidden_dim

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.seq_len, self.hidden_dim * 2),  # 第一层：扩张隐藏维度
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),

            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # 第二层：压缩回原始隐藏维度
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),

            torch.nn.LayerNorm(self.hidden_dim),  # 层归一化
            torch.nn.Linear(self.hidden_dim, self.pred_len)  # 第三层：输出层
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
