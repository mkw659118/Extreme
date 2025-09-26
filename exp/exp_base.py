# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import torch
import os
from time import time
import contextlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer

class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.use_memory = config.use_memory
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
            
            x, x_mark, label, sample_ids = train_batch
            inputs = (x.to(self.config.device), x_mark.to(self.config.device))
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
        
        self.eval()  # 切换为评估模式
        torch.set_grad_enabled(False)  # 关闭梯度计算（评估阶段不需要计算梯度）

        # 选择 dataloader：验证集或测试集
        use_valid = (mode == 'valid') and (len(dataModule.valid_loader.dataset) != 0)
        dataloader = dataModule.valid_loader if use_valid else dataModule.test_loader  # 使用测试集进行评估

        preds, reals, val_loss = [], [], 0.0

        # autocast 设备类型
        device_str = str(self.config.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'

        ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                inputs = (x, x_mark)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()  # 样本 ID 作为输入

                # 前向传播（评估阶段不会写入记忆库）
                pred = self.forward(*inputs, sample_ids=sample_ids)

                # 计算验证集损失
                if use_valid:
                    loss_item = compute_loss(self, inputs, pred, label, self.config)
                    # 若 compute_loss 返回标量张量，下面两行等价；保留求和语义
                    val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                # 存储预测和真实标签
                reals.append(label)
                preds.append(pred)

        # 拼接所有的预测值和真实值
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        # 反归一化处理
        reals = dataModule.y_scaler.inverse_transform(reals.cpu())  # 反归一化真实标签
        preds = dataModule.y_scaler.inverse_transform(preds.cpu())  # 反归一化预测值

        # 学习率调度（只在验证集上进行调度）
        if use_valid and hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.step(val_loss)  # 根据验证集损失调整学习率

        # 当 mode 为 'test' 时，进行预测与真实值的可视化
        if mode == 'test':
            # 获取测试集数据
            x_test, x_mark_test, y_test, sample_ids_test = next(iter(dataModule.test_loader))  # 获取测试集的一个 batch
            x_test = x_test.to(self.config.device)
            x_mark_test = x_mark_test.to(self.config.device)
            y_test = y_test.to(self.config.device)
            sample_ids_test = sample_ids_test.to(self.config.device).long()
            inputs_test = (x_test, x_mark_test)

            # 预测测试集
            preds_test = self.forward(*inputs_test, sample_ids=sample_ids_test)
            x_test = dataModule.y_scaler.inverse_transform(x_test.cpu())  # 反归一化真实标签

            preds_test = dataModule.y_scaler.inverse_transform(preds_test.cpu())  # 反归一化预测值

            # 获取整个测试集的最大值和最小值
            min_val = min(x_test.min(), preds_test.min())
            max_val = max(x_test.max(), preds_test.max())
            mid_val = (min_val + max_val) / 2
            print("调试专用》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》")
            print(min_val)
            print(max_val)

            # 可视化测试集的预测和真实值
            self.visualize_predictions(x_test, preds_test)  # 调用你的可视化函数

        # 返回误差指标
        return ErrorMetrics(reals, preds, self.config)

  
    def visualize_predictions(self, x, y_pred):
        """
        可视化测试集预测值与真实值的对比
        x: 输入数据 [B, L, C]
        y_pred: 预测值 [B, pred_len, 1]
        min_val: 纵坐标最小值
        max_val: 纵坐标最大值
        mid_val: 纵坐标中间值
        """
        # 处理真实值 (x)
        true_values = x[0, -self.pred_len:, 0]  # 获取最后 pred_len 步的真实值
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.cpu().numpy()  # 如果是 Tensor 转换为 numpy

        # 处理预测值 (y_pred)
        predicted_values = y_pred[0]  # 获取最后 pred_len 步的预测值
        if isinstance(predicted_values, torch.Tensor):
            predicted_values = predicted_values.cpu().detach().numpy()  # 如果是 Tensor 转换为 numpy

        # 设置字体（设置 Times New Roman）
        plt.rcParams.update({'font.family': 'Times New Roman'})

        # 设置可视化图像
        plt.figure(figsize=(8, 6))  # 设置图像大小
        plt.plot(true_values, label='True Values', color='tab:blue', linewidth=2)  # 绘制真实值
        plt.plot(predicted_values, label='Predicted Values', linestyle='--', color='tab:orange', linewidth=2)  # 绘制预测值

        # 添加标题和标签
        plt.title('Comparison of Predicted and True Values', fontsize=14)  # 图标题
        plt.xlabel('Time Steps', fontsize=12)  # X轴标签
        plt.ylabel('Values', fontsize=12)  # Y轴标签

        # 自动设置 y 轴范围，以适应数据
        plt.ylim(min(true_values.min(), predicted_values.min()) - 0.1, max(true_values.max(), predicted_values.max()) + 0.1)

        # 添加网格
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # 添加图例
        plt.legend(loc='upper right', fontsize=12)

        # 保存图像到固定路径
        SAVE_DIR = './saved_plots'
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 使用 self.pred_len 和 self.use_memory 生成文件名，并加上时间戳
        file_name = f"prediction_predlen_{self.pred_len}_use_memory_{self.use_memory}_{timestamp}.png"

        save_path = os.path.join(SAVE_DIR, file_name)
        plt.tight_layout()  # 调整布局，避免标签被遮挡
        plt.savefig(save_path, dpi=300)  # 保存为高分辨率图片
        plt.close()  # 关闭图形，避免内存泄漏
        print(f"Plot saved at: {save_path}")


