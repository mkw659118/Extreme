# coding : utf-8
# Author : Yuxiang Zeng
import torch

from modules.temporal_enc import TemporalEmbedding


class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.projection = torch.nn.Linear(1, config.rank)

        self.fund_embedding = torch.nn.Embedding(100000, config.rank)
        self.lstm = torch.nn.LSTM(config.rank, config.rank)
        self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')

        self.fc = torch.nn.Linear(config.rank * config.seq_len, config.pred_len)

    def forward(self, x):
        code_idx = x[:, :, 0].long()
        temporal_idx = x[:, :, 1:5]
        x_seq = x[:, :, -1].unsqueeze(-1)

        x_enc = self.projection(x_seq)
        fund_embeds = self.fund_embedding(code_idx)
        temporal_embeds = self.temporal_embedding(temporal_idx)
        x_embeds, (hn, cn) = self.lstm(x_enc)

        embeds = fund_embeds + temporal_embeds + x_embeds

        y = self.fc(embeds.reshape(embeds.size(0), -1))
        return y