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

        device_str = str(config.device)
        self.device_type = 'cuda' if 'cuda' in device_str else 'cpu'
        self.scaler = torch.amp.GradScaler(config.device)
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

    # =========================================================
    # 将差分域还原回原始域
    # =========================================================
    def _restore_to_raw(self, pred, label, mean, std):
        """
        pred:  [B, L, C_pred]，第0通道=差分(z-score)
        label: [B, L, C_lab]， label[:,0,3]=锚点(i-1 的原值), label[:,:,-1]=原值真值
        mean/std: 训练集“差分”的统计量
        """
        mean_t = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        std_t  = torch.as_tensor(std,  device=pred.device, dtype=pred.dtype)

        pred_diff_z = pred[:, :, 0]
        pred_diff   = pred_diff_z * std_t + mean_t
        y_pre       = label[:, 0, 3]
        pred_raw    = y_pre.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)
        real_raw    = label[:, :, -1]
        return pred_raw, real_raw

    def train_one_epoch(self, dataModule):

        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        total_loss = 0.0
        sample_count = 0

        scaler = self.scaler

        try:
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids = train_batch

                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()

                self.optimizer.zero_grad(set_to_none=True)

                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred = self.forward(x, x_mark, label, sample_ids=sample_ids)

                        pred_raw, real_raw = self._restore_to_raw(
                            pred, label, dataModule.mean, dataModule.std
                        )

                        # 为了数值稳定，loss 用 float32 更稳（尤其是 cumsum 后的序列）
                        loss = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.forward(x, x_mark, label, sample_ids=sample_ids)

                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule.mean, dataModule.std
                    )
                    loss = compute_loss(self, x, pred_raw, real_raw, self.config)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                sample_count += x.size(0)

        except Exception as e:
            print(f"[Train Error] Epoch {self.current_epoch} failed: {str(e)}")

        finally:
            self.eval()
            torch.set_grad_enabled(False)
            t2 = time.time()

            avg_loss = total_loss / sample_count if sample_count > 0 else 0
            print(f"[Train] Epoch {self.current_epoch} finished | "
                  f"Avg Loss: {avg_loss:.6f} | Time Cost: {t2-t1:.2f}s")

        return avg_loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):

        self.eval()
        torch.set_grad_enabled(False)

        use_valid = (mode == 'valid') and (len(dataModule.val_data_loader) != 0)
        dataloader = dataModule.val_data_loader if use_valid else dataModule.test_data_loader

        preds, reals, val_loss = [], [], 0.0

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                pred = self.forward(x, x_mark, label, sample_ids=sample_ids)

                if use_valid:
                    pred_raw, real_raw = self._restore_to_raw(pred, label, dataModule.mean, dataModule.std)
                    loss_item = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                    val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule.mean, dataModule.std)

        # 你原来的评估方式保持：真实值用 reals[:,:,-1]，预测用 pred_raw
        return ErrorMetrics(reals[:, :, -1], pred_raw, self.config, mode)

    def test(self, dataModule, mode='test'):

        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.test_data_loader

        preds, reals = [], []

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                pred = self.forward(x, x_mark, label, sample_ids=sample_ids)

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule.mean, dataModule.std)

        return ErrorMetrics(reals[:, :, -1], pred_raw, self.config, mode)
