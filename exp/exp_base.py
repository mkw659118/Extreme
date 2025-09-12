# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import contextlib
import torch
from time import time

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
from torch.cuda.amp import autocast

class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)  # ✅ 初始化 GradScaler

    def forward(self, *x):
        y = self.model(*x)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if config.classification else 'max', factor=0.5,
                                                                    patience=config.patience // 1.5, threshold=0.0)

    # def train_one_epoch(self, dataModule):
    #     loss = None
    #     self.train()
    #     torch.set_grad_enabled(True)
    #     t1 = time()

    #     for train_batch in dataModule.train_loader:
    #         all_item = [item.to(self.config.device) for item in train_batch]
    #         inputs, label = all_item[:-1], all_item[-1]

    #         self.optimizer.zero_grad()

    #         if self.config.use_amp:
    #             with torch.amp.autocast(device_type=self.config.device):
    #                 pred = self.forward(*inputs)
    #                 loss = compute_loss(self, inputs, pred, label, self.config)

    #             self.scaler.scale(loss).backward()
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         else:
    #             pred = self.forward(*inputs)
    #             loss = compute_loss(self, inputs, pred, label, self.config)
    #             loss.backward()
    #             self.optimizer.step()

    #     t2 = time()
    #     self.eval()
    #     torch.set_grad_enabled(False)
    #     return loss, t2 - t1

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time()

        for train_batch in dataModule.train_loader:
            all_item = [item.to(self.config.device) for item in train_batch]
            inputs, label = all_item[:-1], all_item[-1]

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with torch.amp.autocast(device_type=self.config.device):
                    pred = self.forward(*inputs)
                    # 主损失
                    loss_main = compute_loss(self, inputs, pred, label, self.config)
                    # [ADD] 极端记忆辅助损失（若模型未启用该功能则为0）
                    aux = getattr(self.model, 'last_aux_loss', 0.0)
                    if torch.is_tensor(aux):
                        aux = aux.to(self.config.device)
                    loss = loss_main + aux

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.forward(*inputs)
                # 主损失
                loss_main = compute_loss(self, inputs, pred, label, self.config)
                # [ADD] 极端记忆辅助损失
                aux = getattr(self.model, 'last_aux_loss', 0.0)
                if torch.is_tensor(aux):
                    aux = aux.to(self.config.device)
                loss = loss_main + aux

                loss.backward()
                self.optimizer.step()
            
            if isinstance(loss_main, torch.Tensor) and isinstance(aux, torch.Tensor):
                print(f"[Train] main_loss={loss_main.item():.4f}, aux_loss={aux.item():.6f}")

        t2 = time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1



    def evaluate_one_epoch(self, dataModule, mode='valid'):
        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
        preds, reals, val_loss = [], [], 0.

        context = (
            torch.amp.autocast(device_type=self.config.device)
            if self.config.use_amp else
            contextlib.nullcontext()
        )

        with context:
            for batch in dataloader:
                all_item = [item.to(self.config.device) for item in batch]
                inputs, label = all_item[:-1], all_item[-1]
                pred = self.forward(*inputs)

                if mode == 'valid':
                    # 主验证损失
                    cur = compute_loss(self, inputs, pred, label, self.config)
                    # [ADD] 同步计入辅助损失（不影响测试，仅在 valid 阶段用于调度/early stopping）
                    aux = getattr(self.model, 'last_aux_loss', 0.0)
                    if torch.is_tensor(aux):
                        aux = aux.detach()
                    val_loss += (cur + aux)

                if self.config.classification:
                    pred = torch.max(pred, 1)[1]

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        # if self.config.dataset != 'weather':
        reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)

        if mode == 'valid':
            self.scheduler.step(val_loss)

        return ErrorMetrics(reals, preds, self.config)



    # def evaluate_one_epoch(self, dataModule, mode='valid'):
    #     self.eval()
    #     torch.set_grad_enabled(False)
    #     dataloader = dataModule.valid_loader if mode == 'valid' and len(dataModule.valid_loader.dataset) != 0 else dataModule.test_loader
    #     preds, reals, val_loss = [], [], 0.

    #     context = (
    #         torch.amp.autocast(device_type=self.config.device)
    #         if self.config.use_amp else
    #         contextlib.nullcontext()
    #     )

    #     with context:
    #         for batch in dataloader:
    #             all_item = [item.to(self.config.device) for item in batch]
    #             inputs, label = all_item[:-1], all_item[-1]
    #             pred = self.forward(*inputs)

    #             if mode == 'valid':
    #                 val_loss += compute_loss(self, inputs, pred, label, self.config)

    #             if self.config.classification:
    #                 pred = torch.max(pred, 1)[1]

    #             reals.append(label)
    #             preds.append(pred)

    #     reals = torch.cat(reals, dim=0)
    #     preds = torch.cat(preds, dim=0)

    #     # if self.config.dataset != 'weather':
    #     reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)

    #     if mode == 'valid':
    #         self.scheduler.step(val_loss)

    #     return ErrorMetrics(reals, preds, self.config)