# coding : utf-8
# Author : Yuxiang Zeng
import torch 
import torch.nn.functional as F

# 在这里加上每个Batch的loss，如果有其他的loss，请在这里添加，
def compute_loss(model, inputs, pred, label, config):
    loss = model.loss_function(pred, label)
    return loss
