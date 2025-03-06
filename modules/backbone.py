# coding : utf-8
# Author : Yuxiang Zeng
import torch

class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_layer = torch.nn.Linear(enc_in * config.seq_len, config.pred_len)


    def forward(self, x):
        y = self.pred_layer(x.reshape(x.size(0), -1))
        return y