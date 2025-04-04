# coding : utf-8
# Author : Yuxiang Zeng
import torch

from baselines.timesnet import TimesBlock
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.revin import RevIN
from layers.transformer import Transformer
from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft
        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)

        # self.projection = torch.nn.Linear(1, config.rank, bias=True)
        self.projection = TokenEmbedding(1, config.rank)

        self.position_embedding = PositionEncoding(d_model=self.rank, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(999999, config.rank)
        self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)
        self.encoder = Transformer(
            self.rank,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )
        # self.encoder = torch.nn.ModuleList([TimesBlock(config) for _ in range(config.num_layers)])
        # self.layer_norm = torch.nn.LayerNorm(self.rank)
        self.fc = torch.nn.Linear(config.rank, 1)

    def forward(self, x, x_mark):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        x = x.unsqueeze(-1)

        # print(x.shape, x_mark.shape)
        x_enc = self.projection(x)
        x_enc += self.position_embedding(x_enc)
        # x_enc += self.tempora l_embedding(x_mark)
        # x_enc += self.fund_embedding(code_idx)
        x_enc = self.predict_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        x_enc = self.encoder(x_enc)
        # for i in range(len(self.encoder)):
        #     x_enc = self.layer_norm(self.encoder[i](x_enc))

        # x_enc += torch.cat([self.fund_embedding(code_idx), self.fund_embedding(code_idx)], dim=1)

        x_enc = self.fc(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]
        # y = self.fc(x_enc.reshape(x_enc.size(0), -1))

        # denorm
        if self.revin:
            y = self.revin_layer(y, 'denorm')
        return y