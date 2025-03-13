# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.encoder.seq_enc import SeqEncoder
from layers.transformer import Transformer
from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.projection = torch.nn.Linear(1, config.rank)

        # self.fund_embedding = torch.nn.Embedding(100000, config.rank)
        # self.seq_encoder = SeqEncoder(input_size=config.rank, d_model=config.rank, seq_len=config.seq_len, num_layers=config.num_layers, seq_method='gru', bidirectional=True)
        # self.lstm = torch.nn.LSTM(config.rank, config.rank)
        # self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')

        self.encoder = Transformer(self.rank, num_heads=4, num_layers=4, norm_method='layer', ffn_method='ffn', att_method='self')
        self.fc = torch.nn.Linear(config.rank * config.seq_len, config.pred_len)

    def forward(self, x):
        code_idx = x[:, :, 0].long()
        temporal_idx = x[:, :, 1:5]
        x_seq = x[:, :, -1].unsqueeze(-1)

        x_enc = self.projection(x_seq)
        x_enc = self.encoder(x_enc)

        # embeds, (hn, cn) = self.lstm(x_enc)
        # embeds = self.seq_encoder(x_enc)
        # embeds += self.fund_embedding(code_idx)
        # embeds += self.temporal_embedding(temporal_idx)
        y = self.fc(x_enc.reshape(x_enc.size(0), -1))

        return y