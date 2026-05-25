import numpy as np
import pandas as pd
import torch
import time
import contextlib
from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
import os
from tqdm import *
import pickle
from utils.model_monitor import EarlyStopping


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.pred_len = config.pred_len
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
        
    def save_single_model_result(self, true_series, pred_series, save_path):
        if isinstance(true_series, torch.Tensor):
            true_series = true_series.detach().cpu().numpy()
        else:
            true_series = np.asarray(true_series)

        if isinstance(pred_series, torch.Tensor):
            pred_series = pred_series.detach().cpu().numpy()
        else:
            pred_series = np.asarray(pred_series)

        true_series = true_series.reshape(-1)
        pred_series = pred_series.reshape(-1)

        assert len(true_series) == len(pred_series), \
            f"true 和 pred 长度不一致: {len(true_series)} vs {len(pred_series)}"

        df = pd.DataFrame({
            "index": np.arange(len(true_series)),
            "true": true_series,
            "pred": pred_series,
        })

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"[Save] single model result saved to: {save_path}")

    def RunOnce(self, config, runId, model, datamodule, log):
        try:
            # 一些模型（如Keras兼容模型）可能需要compile，跳过非必要的compile
            model.compile()
        except Exception as e:
            print(f'Skip the model.compile() because {e}')

        # 设置EarlyStopping监控器
        monitor = EarlyStopping(config)

        # 创建保存模型的目录
        os.makedirs(f'./checkpoints/{config.model}', exist_ok=True)
        model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'

        # 判断是否需要重新训练：
        # 若 config.retrain==1 表示强制重训；
        # 或者模型文件不存在 且 设置了 continue_train，则需要重新训练
        retrain_required = config.retrain == 1 or not os.path.exists(model_path) and config.continue_train

        # 如果无需重新训练且已有模型文件，则直接加载模型并评估测试集性能
        if not retrain_required:
            try:
                # 加载之前记录的训练时间
                sum_time = pickle.load(open(f'./results/metrics/' + log.filename + '.pkl', 'rb'))['train_time'][runId]
                # 加载模型权重（weights_only=True 可忽略 optimizer 等无关信息）
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
                model.setup_optimizer(config)  # 重新设置优化器
                results = model.test(datamodule)  # 在测试集评估性能
                log.show_results(results, sum_time)
                config.record = False  # 不再记录当前结果
            except Exception as e:
                log.only_print(f'Error: {str(e)}')
                retrain_required = True  # 若加载失败则触发重新训练

        # 若设置为继续训练（即接着上次的结果继续）
        if config.continue_train:
            log.only_print(f'Continue training...')
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

        # 若需要重新训练
        if retrain_required:
            model.setup_optimizer(config)
            train_time = []
            for epoch in trange(config.epochs):
                model.current_epoch = epoch + 1  # 从1开始计数，更符合实际习惯

                if monitor.early_stop:
                    break  # 若满足early stopping条件则提前终止训练

                # 训练一个epoch并记录耗时
                train_loss, time_cost = model.train_one_epoch(datamodule)
                train_time.append(time_cost)

                # 验证集上评估当前模型误差
                valid_error = model.valid(datamodule)

                model.scheduler.step(valid_error[config.monitor_metric])
                print(f"Current LR: {model.optimizer.param_groups[0]['lr']:.6g}")

                # 将当前epoch的验证误差传递给early stopping模块进行跟踪
                monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

                # 输出当前epoch的训练误差和验证误差，并记录训练时间
                log.show_epoch_error(runId, epoch, monitor, train_loss, valid_error, train_time)

                # 更新日志可视化（如绘图）
                log.plotter.append_epochs(train_loss, valid_error)

                # 暂存模型参数（即使不是最优，也为了中断续训做准备）
                torch.save(model.state_dict(), model_path)

            # 加载最优模型参数（来自early stopping）
            model.load_state_dict(monitor.best_model)

            # 累计训练时间（仅使用前best_epoch轮）
            sum_time = sum(train_time[: monitor.best_epoch])

            # 使用最优模型在测试集评估
            results = model.test(datamodule)
        
            log.show_test_error(runId, monitor, results, sum_time)

            # 保存最优模型参数
            torch.save(monitor.best_model, model_path)
            log(f'Model parameters saved to {model_path}')

        # 将训练时间加入返回结果中
        results['train_time'] = sum_time
        
        return results

    
    # 将多变量标准化差分还原回原始域
    def _restore_to_raw(self, pred, label, dataModule, sample_ids):
        mean_t = torch.as_tensor(
            dataModule.mean,
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, 1, -1)

        std_t = torch.as_tensor(
            dataModule.std,
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, 1, -1)

        raw_data_t = torch.as_tensor(
            dataModule.raw_data,
            device=pred.device,
            dtype=pred.dtype,
        )

        sample_ids = torch.as_tensor(
            sample_ids,
            device=pred.device,
            dtype=torch.long,
        )

        pred_diff = pred * std_t + mean_t
        real_diff = label * std_t + mean_t

        anchor_index = sample_ids + self.config.seq_len - 1
        anchor = raw_data_t[anchor_index]

        pred_raw = anchor.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)
        real_raw = anchor.unsqueeze(1) + torch.cumsum(real_diff, dim=1)

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
                # sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()

                self.optimizer.zero_grad(set_to_none=True)

                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred = self.forward(x, x_mark, sample_ids=sample_ids, mode='train')

                        pred_raw, real_raw = self._restore_to_raw(
                            pred, label, dataModule, sample_ids
                        )

                        # 为了数值稳定，loss 用 float32 更稳（尤其是 cumsum 后的序列）
                        loss = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.forward(x, x_mark, sample_ids=sample_ids, mode='train')

                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule, sample_ids
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

    def valid(self, dataModule):

        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader 

        preds, reals, ids, val_loss = [], [], [], 0.0

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                pred = self.forward(x, x_mark, sample_ids=sample_ids, mode='valid')

                pred_raw, real_raw = self._restore_to_raw(pred, label, dataModule, sample_ids)
                loss_item = compute_loss(self, x, pred_raw.float(), real_raw.float(), self.config)
                val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                reals.append(label)
                preds.append(pred)
                ids.append(sample_ids)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        ids = torch.cat(ids, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule, ids)

        return ErrorMetrics(real_raw, pred_raw, self.config, 'valid')

    def test(self, dataModule):

        self.eval()
        torch.set_grad_enabled(False)
        dataloader = dataModule.test_data_loader

        preds, reals, ids = [], [], []

        ctx = torch.autocast(device_type=self.device_type, dtype=torch.float16) if self.config.use_amp else contextlib.nullcontext()

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch
                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                pred = self.forward(x, x_mark, sample_ids=sample_ids, mode='test')

                reals.append(label)
                preds.append(pred)
                ids.append(sample_ids)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        ids = torch.cat(ids, dim=0)

        pred_raw, real_raw = self._restore_to_raw(preds, reals, dataModule, ids)
        # 保存测试集真实序列和预测序列用于画图
        pred_series = pred_raw.reshape(-1)
        true_series = real_raw.reshape(-1)

        self.save_single_model_result(
            true_series=true_series,
            pred_series=pred_series,
            save_path=f"./draw/{self.config.model}_{self.config.reservoir_sensor}_PL{self.config.pred_len}_DM{self.config.d_model}.csv"
        )

        return ErrorMetrics(real_raw, pred_raw, self.config, 'test')
