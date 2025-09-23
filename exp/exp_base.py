# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import torch
from time import time

import matplotlib.pyplot as plt


from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer

class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.scaler = torch.amp.GradScaler(config.device)  # ✅ 初始化 GradScaler

    def forward(self, *x, **kwargs):
        y = self.model(*x, **kwargs)
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

    #         if self.config.use_amp: # 是否启用混合精度训练
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

        # 识别 autocast 的 device_type
        device_str = str(self.config.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'

        for train_batch in dataModule.train_loader:
            # -------- 解包 batch：支持 (x, x_mark, y, ids) 或 (x, y, ids) --------
            if len(train_batch) == 4:
                x, x_mark, label, sample_ids = train_batch
                inputs = (x.to(self.config.device), x_mark.to(self.config.device))
            elif len(train_batch) == 3:
                x, label, sample_ids = train_batch
                inputs = (x.to(self.config.device),)
            else:
                raise ValueError(f"Unexpected train_batch size: {len(train_batch)}")

            label = label.to(self.config.device)
            sample_ids = sample_ids.to(self.config.device).long()  # [B] LongTensor

            self.optimizer.zero_grad()

            if self.config.use_amp:
                # 你也可以改成 dtype=torch.bfloat16 视显卡支持情况而定
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = self.forward(*inputs, sample_ids=sample_ids)
                    loss = compute_loss(self, inputs, pred, label, self.config)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.forward(*inputs, sample_ids=sample_ids)
                loss = compute_loss(self, inputs, pred, label, self.config)
                loss.backward()
                self.optimizer.step()

        t2 = time()
        self.eval()
        torch.set_grad_enabled(False)

        return loss, t2 - t1


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

    #     reals, preds = dataModule.y_scaler.inverse_transform(reals), dataModule.y_scaler.inverse_transform(preds)

    #     if mode == 'valid':
    #         self.scheduler.step(val_loss)

    #     return ErrorMetrics(reals, preds, self.config)

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        import contextlib

        self.eval()
        torch.set_grad_enabled(False)

        # 选择 dataloader
        use_valid = (mode == 'valid') and (len(dataModule.valid_loader.dataset) != 0)
        dataloader = dataModule.valid_loader if use_valid else dataModule.test_loader

        preds, reals, val_loss = [], [], 0.0

        # autocast 设备类型
        device_str = str(self.config.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'

        ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                # -------- 解包 batch：支持 (x, x_mark, y, ids) 或 (x, y, ids) --------
                if len(batch) == 4:
                    x, x_mark, label, sample_ids = batch
                    x = x.to(self.config.device)
                    x_mark = x_mark.to(self.config.device)
                    inputs = (x, x_mark)
                elif len(batch) == 3:
                    x, label, sample_ids = batch
                    x = x.to(self.config.device)
                    inputs = (x,)
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()  # [B]

                # 前向（评估阶段不会写入记忆库）
                pred = self.forward(*inputs, sample_ids=sample_ids)

                # 验证集计算 loss（可求标量或张量累计，视你的 compute_loss 返回类型）
                if use_valid:
                    loss_item = compute_loss(self, inputs, pred, label, self.config)
                    # 若 compute_loss 返回标量张量，下面两行等价；保留求和语义
                    val_loss = val_loss + (loss_item.item() if torch.is_tensor(loss_item) else float(loss_item))

                # 分类任务：取类别索引
                if getattr(self.config, "classification", False):
                    pred = pred.argmax(dim=1)

                reals.append(label)
                preds.append(pred)

        # 拼接
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        # 反归一化（若你的 y_scaler 期望 numpy，可在内部做 .cpu().numpy()）
        reals = dataModule.y_scaler.inverse_transform(reals)
        preds = dataModule.y_scaler.inverse_transform(preds)

        # 学习率调度（只在 valid 上）
        if use_valid and hasattr(self, "scheduler") and self.scheduler is not None:
            # 若你的 scheduler 期望的是平均 loss，可改为 val_loss / len(dataloader)
            self.scheduler.step(val_loss)

        plot_all_test_results(reals, preds, save_path="pictures/test_results.png")

        return ErrorMetrics(reals, preds, self.config)


def plot_all_test_results(reals, preds, save_path=None):
    """
    绘制整个测试集的真实值 vs 预测值
    reals: torch.Tensor 或 np.ndarray, shape [N, pred_len, C] 或 [N, C]
    preds: torch.Tensor 或 np.ndarray, shape [N, pred_len, C] 或 [N, C]
    save_path: 如果指定，则保存到文件；否则直接 plt.show()
    """
    # 转 numpy
    if torch.is_tensor(reals):
        reals = reals.detach().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    
    # 保证是二维 [N, pred_len]
    reals = reals.squeeze()
    preds = preds.squeeze()
    
    # 如果是 [N, pred_len]，拼接成 1D 长序列
    if reals.ndim == 2:
        reals = reals.reshape(-1)
        preds = preds.reshape(-1)

    plt.figure(figsize=(20, 5))
    plt.plot(reals, label="Real", color="black")
    plt.plot(preds, label="Pred", color="red", alpha=0.7, linestyle="--")
    plt.title("Test set: Real vs Predicted")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved test plot to {save_path}")
    else:
        plt.show()
