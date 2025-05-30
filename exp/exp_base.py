# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import torch
from time import time

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config

    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if config.classification else 'max', factor=0.5,
                                                                    patience=config.patience // 1.5, threshold=0.0)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time()
        for train_batch in (dataModule.train_loader):
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]
            pred = self.forward(*inputs)
            loss = compute_loss(self, pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
        preds, reals, val_loss = [], [], 0.
        for batch in (dataloader):
            all_item = [item.to(self.config.device) for item in batch]
            inputs, label = all_item[:-1], all_item[-1]
            pred = self.forward(*inputs)
            if mode == 'valid':
                val_loss += self.loss_function(pred, label)
            if self.config.classification:
                pred = torch.max(pred, 1)[1]
            reals.append(label)
            preds.append(pred)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        if self.config.dataset != 'weather':
            reals, preds = dataModule.scaler.inverse_transform(reals), dataModule.scaler.inverse_transform(preds)
        if mode == 'valid':
            self.scheduler.step(val_loss)
        metrics_error = ErrorMetrics(reals, preds, self.config)
        return metrics_error