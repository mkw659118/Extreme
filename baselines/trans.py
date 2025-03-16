# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.transformer import Transformer
from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        # self.projection = torch.nn.Linear(1, config.rank)
        self.projection = TokenEmbedding(1, config.rank)
        self.position_embedding = PositionEncoding(d_model=self.rank, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(100000, config.rank)
        self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')

        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)
        self.encoder = Transformer(self.rank, num_heads=8, num_layers=12, norm_method=config.norm_method, ffn_method=config.ffn_method, att_method=config.att_method)
        self.fc = torch.nn.Linear(config.rank, 1)

    def forward(self, x):
        x_enc = self.projection(x)
        x_enc += self.position_embedding(x_enc)

        x_enc = self.predict_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        x_enc = self.encoder(x_enc)
        # y = self.fc(x_enc.reshape(x_enc.size(0), -1))
        x_enc = self.fc(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]
        # y = self.fc(x_enc.reshape(x_enc.size(0), -1))
        return y