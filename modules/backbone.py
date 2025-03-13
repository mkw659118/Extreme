# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.transformer import Transformer
from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.projection = torch.nn.Linear(1, config.rank)

        self.position_embedding = PositionEncoding(d_model=self.rank, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(100000, config.rank)
        self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')

        # self.lstm = torch.nn.LSTM(config.rank, config.rank)
        # self.encoder = SeqEncoder(input_size=config.rank, d_model=config.rank, seq_len=config.seq_len, num_layers=config.num_layers, seq_method='gru', bidirectional=True)
        self.encoder = Transformer(self.rank, num_heads=8, num_layers=16, norm_method='rms', ffn_method='moe', att_method='self')

        self.fc = torch.nn.Linear(config.rank * config.seq_len, config.pred_len)


    def forward(self, x):
        # code_idx = x[:, :, 0].long()
        temporal_idx = x[:, :, 1:5]
        x_seq = x[:, :, -1].unsqueeze(-1)

        x_enc = self.projection(x_seq)
        x_enc += self.temporal_embedding(temporal_idx)
        x_enc += self.position_embedding(x_enc)

        x_enc = self.encoder(x_enc)

        # x_enc += self.fund_embedding(code_idx)
        y = self.fc(x_enc.reshape(x_enc.size(0), -1))

        return y