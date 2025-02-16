# coding : utf-8
# Author : Yuxiang Zeng
import torch

class Backbone(torch.nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_layer = torch.nn.Linear(4, 20)


    def forward(self, x):
        y = self.pred_layer(x)
        return y