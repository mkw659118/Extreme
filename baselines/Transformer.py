# coding: utf-8
# Author: mkw
# Date: 2025-06-04 17:40
# Description: Transformer

from einops import rearrange
from torch import nn
from torchvision.models.video.mvit import PositionalEncoding

from layers.revin import RevIN


class TransformerModel(nn.Module):
    def __init__(self, enc_in, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.revin = config.revin
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.d_model = config.d_model

        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        self.input_projection = nn.Linear(enc_in, self.d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=getattr(config, "positional_encoding_max_len", 5000)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)

        self.output_layer = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, x_mark=None):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        x = self.input_projection(x)  # [bs, seq_len, d_model]
        x = rearrange(x, 'bs seq_len d_model -> seq_len bs d_model')
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = rearrange(x, 'seq_len bs d_model -> bs d_model seq_len')

        y = self.output_layer(x)
        y = rearrange(y, 'bs d_model pred_len -> bs pred_len d_model')

        if self.revin:
            y = self.revin_layer(y, 'denorm')
        return y
