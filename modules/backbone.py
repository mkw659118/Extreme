# coding : utf-8
# Author : Yuxiang Zeng
import torch
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.revin import RevIN
from layers.transformer import Transformer


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft


        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)

        self.projection = torch.nn.Linear(enc_in, config.d_model, bias=True)
        self.position_embedding = PositionEncoding(d_model=self.d_model, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(999999, config.d_model)
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.encoder1 = Transformer(
            self.d_model,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )

        self.encoder2 = Transformer(
            self.d_model,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )

        # self.encoder1 = torch.nn.Linear(config.d_model, config.d_model, bias=True)
        # self.encoder2 = torch.nn.Linear(config.d_model, config.d_model, bias=True)

        self.decoder = torch.nn.Linear(config.d_model, 3)

    def forward(self, x, x_mark, x_fund):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x_enc = self.projection(x)
        # x_enc = self.fund_embedding(x_fund)
        x_enc = self.predict_linear(x_enc.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        bs, pred_len, channels, dim = x_enc.shape
        x_enc = x_enc.permute(2, 0, 1, 3).reshape(channels, bs * pred_len, dim)
        x_enc = self.encoder1(x_enc)
        x_enc = x_enc.reshape(channels, bs, pred_len, dim).permute(1, 2, 0, 3)
        x_enc = x_enc.permute(1, 0, 2, 3).reshape(pred_len, bs * channels, dim)
        x_enc = self.encoder2(x_enc)
        x_enc = x_enc.reshape(pred_len, bs, channels, dim).permute(1, 0, 2, 3)

        x_enc = self.decoder(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)
        
        if self.revin:
            y = self.revin_layer(y, 'denorm')
        return y
