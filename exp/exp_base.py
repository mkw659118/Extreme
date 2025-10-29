# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年3月27日23:33:32）
import torch
import time
import contextlib
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
        self.current_epoch = 0  # 从0开始计数，第1个epoch训练时会更新为1
        
    def forward(self, *x, **kwargs):
        y = self.model(*x, **kwargs)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(self.parameters(), lr=config.lr, decay=config.decay, config=config)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 5,
            threshold=1e-3
        )

    # 将差分域还原回原始域
    def _restore_to_raw(self, pred, label, mean, std):
        """
        pred:  [B, L, C_pred]，第0通道=差分(z-score)
        label: [B, L, C_lab]， label[:,0,3]=锚点(i-1 的原值), label[:,:,-1]=原值真值
        mean/std: 训练集“差分”的统计量
        """
        # 转成与 pred 同 device / dtype，避免 AMP 下类型不匹配
        mean_t = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        std_t  = torch.as_tensor(std,  device=pred.device, dtype=pred.dtype)

        pred_diff_z = pred[:, :, 0]                 # 差分(z-score)
        pred_diff   = pred_diff_z * std_t + mean_t  # 差分(原尺度)
        y_pre       = label[:, 0, 3]                # 锚点(i-1 原值), [B]
        pred_raw    = y_pre.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)  # [B, L]
        real_raw    = label[:, :, -1]               # 真值原尺度, [B, L]
        return pred_raw, real_raw


    def train_one_epoch(self, dataModule):
        
        self.train()  # 设置模型为训练模式
        torch.set_grad_enabled(True)
        t1 = time.time()
        
        # 初始化损失累加器和样本计数
        total_loss = 0.0
        sample_count = 0
        
        # 识别设备类型（用于AMP）
        device_str = str(self.config.device)
        device_type = 'cuda' if 'cuda' in device_str else 'cpu'
        
        # 初始化混合精度缩放器
        scaler = self.scaler
        
        try:
            # 遍历训练数据加载器
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids = train_batch
                
                # 数据转移至目标设备
                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()
                
                # 梯度清零
                self.optimizer.zero_grad(set_to_none=True)
                
                # 混合精度训练流程
                if self.config.use_amp:
                    with torch.autocast(device_type=device_type, dtype=torch.float16):
                        # 前向传播
                        pred = self.forward(x, x_mark, sample_ids=sample_ids)
                        # 计算损失（需确保compute_loss兼容AMP）
                        loss = compute_loss(self, x, pred, label, self.config)
                    
                    # 反向传播（带梯度缩放）
                    scaler.scale(loss).backward()
                    # 先反缩放，再裁剪
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # 普通训练流程
                    pred = self.forward(x, x_mark, sample_ids=sample_ids)
                    # loss = compute_loss(self, x, pred, label, self.config)
                    # === 新增：原尺度还原 + 原尺度 loss ===
                    pred_raw, real_raw = self._restore_to_raw(pred, label, dataModule.mean, dataModule.std)
                    loss = compute_loss(self, x, pred_raw, real_raw, self.config)      # 原尺度损失
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # 累加损失（使用item()避免计算图残留）
                total_loss += loss.item() * x.size(0)
                sample_count += x.size(0)
                
        except Exception as e:
            print(f"[Train Error] Epoch {self.current_epoch} failed: {str(e)}")
        
        finally:
            # 结束训练，设置模型为评估模式
            self.eval()
            torch.set_grad_enabled(False)
            t2 = time.time()
            
            # 计算平均损失
            avg_loss = total_loss / sample_count if sample_count > 0 else 0
            print(f"[Train] Epoch {self.current_epoch} finished | "
                f"Avg Loss: {avg_loss:.6f} | Time Cost: {t2-t1:.2f}s")
        
        return avg_loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        
        self.eval()  # 切换为评估模式
        torch.set_grad_enabled(False)  # 关闭梯度计算（评估阶段不需要计算梯度）

        # 选择 dataloader：验证集或测试集
        use_valid = (mode == 'valid') and (len(dataModule.val_data_loader) != 0)
        dataloader = dataModule.val_data_loader if use_valid else dataModule.test_data_loader  # 使用测试集进行评估

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
                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule.mean, dataModule.std
                    )
                    loss_item = compute_loss(self, x, pred_raw, real_raw, self.config)      # 原尺度损失
                    # 若 compute_loss 返回标量张量，下面两行等价；保留求和语义
                    val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                # 存储预测和真实标签
                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)   
        preds = torch.cat(preds, dim=0)
    
        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule.mean, dataModule.std)

        return ErrorMetrics(reals[:, :, -1], pred_raw, self.config)