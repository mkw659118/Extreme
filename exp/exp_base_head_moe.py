import os
import time
import pickle
import contextlib

import torch
from tqdm import trange

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_trainer import get_loss_function, get_optimizer
from utils.model_monitor import EarlyStopping


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.label_len = config.label_len

        device_str = str(config.device)
        self.device_type = 'cuda' if 'cuda' in device_str else 'cpu'

        # AMP scaler
        self.scaler = torch.amp.GradScaler(config.device)

        # 当前 epoch，便于日志打印
        self.current_epoch = 0

    def forward(self, *x, **kwargs):
        y = self.model(*x, **kwargs)
        return y

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)
        self.optimizer = get_optimizer(
            self.parameters(),
            lr=config.lr,
            decay=config.decay,
            config=config
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 5,
            threshold=1e-3
        )

    def RunOnce(self, config, runId, model, datamodule, log):
        try:
            # 某些模型可能有 compile，若无则跳过
            model.compile()
        except Exception as e:
            print(f"Skip the model.compile() because {e}")

        monitor = EarlyStopping(config)

        os.makedirs(f'./checkpoints/{config.model}', exist_ok=True)
        model_path = f'./checkpoints/{config.model}/{log.filename}_round_{runId}.pt'

        retrain_required = (
            config.retrain == 1
            or (not os.path.exists(model_path) and config.continue_train)
        )

        # --------------------------------------------------
        # 直接加载已有模型并测试
        # --------------------------------------------------
        if not retrain_required:
            try:
                sum_time = pickle.load(
                    open(f'./results/metrics/' + log.filename + '.pkl', 'rb')
                )['train_time'][runId]

                model.load_state_dict(
                    torch.load(model_path, weights_only=True, map_location='cpu')
                )
                model.setup_optimizer(config)

                results = model.test(datamodule)
                log.show_results(results, sum_time)
                config.record = False

            except Exception as e:
                log.only_print(f'Error: {str(e)}')
                retrain_required = True

        # --------------------------------------------------
        # 继续训练
        # --------------------------------------------------
        if config.continue_train:
            log.only_print('Continue training...')
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location='cpu')
            )

        # --------------------------------------------------
        # 重新训练
        # --------------------------------------------------
        if retrain_required:
            model.setup_optimizer(config)
            train_time = []

            for epoch in trange(config.epochs):
                model.current_epoch = epoch + 1

                if monitor.early_stop:
                    break

                train_loss, time_cost = model.train_one_epoch(datamodule)
                train_time.append(time_cost)

                valid_error = model.valid(datamodule)

                model.scheduler.step(valid_error[config.monitor_metric])
                print(f"Current LR: {model.optimizer.param_groups[0]['lr']:.6g}")

                monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

                log.show_epoch_error(runId, epoch, monitor, train_loss, valid_error, train_time)
                log.plotter.append_epochs(train_loss, valid_error)

                # 每个 epoch 暂存
                torch.save(model.state_dict(), model_path)

            # 加载最优参数
            model.load_state_dict(monitor.best_model)

            # 构建 retrieval index
            train_data = datamodule.train_data_loader.dataset
            train_loader = datamodule.train_data_loader
            model.prepare_retrieval_index(train_data, train_loader)

            # 仅累计 best_epoch 之前的训练时间
            sum_time = sum(train_time[: monitor.best_epoch])

            results = model.test(datamodule)
            log.show_test_error(runId, monitor, results, sum_time)

            torch.save(monitor.best_model, model_path)
            log(f'Model parameters saved to {model_path}')

        results['train_time'] = sum_time
        return results

    # =========================================================
    # 1. 将“预测的差分域”还原回“原始值域”
    #
    # 新版 label 列定义：
    #   label[:, :, 0] -> 核心归一化值（未来 diff 的归一化目标）
    #   label[:, :, 1] -> 差分锚点列（前一时刻原始值）
    #   label[:, :, 2] -> 原始值列（当前时刻真实原始值）
    # =========================================================
    def _restore_to_raw(self, pred, label, mean, std):
        mean_t = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        std_t = torch.as_tensor(std, device=pred.device, dtype=pred.dtype)

        # pred: [B, pred_len, 1]
        pred_diff_z = pred[:, :, 0]          # 预测的归一化差分
        pred_diff = pred_diff_z * std_t + mean_t

        # 第一个预测步之前的原始锚点
        y_pre = label[:, 0, 1]

        # 差分累加还原
        pred_raw = y_pre.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)

        # 真实原始值
        real_raw = label[:, :, 2]

        return pred_raw, real_raw

    # =========================================================
    # 2. 构建 retrieval index
    #    注意：dataset 现在返回 5 个值
    # =========================================================
    def prepare_retrieval_index(self, train_data, train_loader):
        print('*******Constructing the Retrieval Indexes*********')
        time_now = time.time()
        train_steps = len(train_loader)

        self.model.construct_index(len(train_data))

        with torch.no_grad():
            for epoch in range(1):
                iter_count = 0

                for i, batch in enumerate(train_loader):
                    batch_x, x_mark, batch_y, sample_ids, route_labels = batch

                    batch_x = batch_x.float().to(self.config.device)
                    batch_y = batch_y.float().to(self.config.device)
                    sample_ids = sample_ids.to(self.config.device).long()

                    iter_count += 1

                    # value 存储的是 label 的最后 pred_len 段
                    self.model.add_key_value(
                        batch_x,
                        batch_y[:, -self.config.pred_len:, :],
                        sample_ids
                    )

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((1 - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        print('*******Finishing the Retrieval Indexes*********')
        self.model.value_permute = self.model.values.permute(2, 0, 1)

    # =========================================================
    # 3. 训练一个 epoch
    #    现在 forward 返回:
    #       pred, aux_loss
    #    其中 aux_loss 已经包含:
    #       balance_loss + route_loss_weight * route_loss
    # =========================================================
    def train_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        total_loss = 0.0
        sample_count = 0

        scaler = self.scaler

        try:
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids, route_labels = train_batch

                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()
                route_labels = route_labels.to(self.config.device, non_blocking=True).long()

                # decoder input
                # label 现在是 [B, pred_len, 3]，但仍然沿用你原来的 decoder 输入构造方式
                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1
                ).float().to(self.config.device)

                self.optimizer.zero_grad(set_to_none=True)

                # ---------------- AMP 分支 ----------------
                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred, aux_loss = self.forward(
                            x,
                            x_mark,
                            dec_input=dec_input,
                            sample_ids=sample_ids,
                            route_labels=route_labels,
                            mode='train'
                        )

                        pred_raw, real_raw = self._restore_to_raw(
                            pred, label, dataModule.mean, dataModule.std
                        )

                        main_loss = compute_loss(
                            self,
                            x,
                            pred_raw.float(),
                            real_raw.float(),
                            self.config
                        )

                        total_loss_value = main_loss + aux_loss

                    scaler.scale(total_loss_value).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()

                # ---------------- 非 AMP 分支 ----------------
                else:
                    pred, aux_loss = self.forward(
                        x,
                        x_mark,
                        dec_input=dec_input,
                        sample_ids=sample_ids,
                        route_labels=route_labels,
                        mode='train'
                    )

                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule.mean, dataModule.std
                    )

                    main_loss = compute_loss(self, x, pred_raw, real_raw, self.config)
                    total_loss_value = main_loss + aux_loss

                    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataModule.train_data_loader):
                        route_loss_item = 0.0
                        if hasattr(self.model, "latest_aux_dict") and "route_loss" in self.model.latest_aux_dict:
                            route_loss_item = self.model.latest_aux_dict["route_loss"].item()

                        print(
                            f"Batch {batch_idx+1}/{len(dataModule.train_data_loader)} - "
                            f"Main Loss: {main_loss.item():.6f}, "
                            f"Aux Loss: {aux_loss.item():.6f}, "
                            f"Route Loss: {route_loss_item:.6f}, "
                            f"Total Loss: {total_loss_value.item():.6f}"
                        )

                    total_loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += total_loss_value.item() * x.size(0)
                sample_count += x.size(0)

        except Exception as e:
            print(f"[Train Error] Epoch {self.current_epoch} failed: {str(e)}")

        finally:
            self.eval()
            torch.set_grad_enabled(False)
            t2 = time.time()

            avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
            print(
                f"[Train] Epoch {self.current_epoch} finished | "
                f"Avg Loss: {avg_loss:.6f} | Time Cost: {t2 - t1:.2f}s"
            )

        return avg_loss, t2 - t1

    # =========================================================
    # 4. 验证
    #    验证集 batch 现在也按 5 个值解包
    #    但 route_labels 不参与验证监督，可直接忽略
    # =========================================================
    def valid(self, dataModule):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader

        preds, reals = [], []
        val_loss = 0.0

        ctx = (
            torch.autocast(device_type=self.device_type, dtype=torch.float16)
            if self.config.use_amp else contextlib.nullcontext()
        )

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids, route_labels = batch

                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1
                ).float().to(self.config.device)

                pred, _ = self.forward(
                    x,
                    x_mark,
                    dec_input=dec_input,
                    sample_ids=sample_ids,
                    route_labels=None,
                    mode='valid'
                )

                pred_raw, real_raw = self._restore_to_raw(
                    pred, label, dataModule.mean, dataModule.std
                )

                loss_item = compute_loss(
                    self,
                    x,
                    pred_raw.float(),
                    real_raw.float(),
                    self.config
                )
                val_loss += loss_item.item() if torch.is_tensor(loss_item) else float(loss_item)

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(
            preds, reals, dataModule.mean, dataModule.std
        )

        # 评估仍基于原始值域
        return ErrorMetrics(reals[:, :, 2], pred_raw, self.config, 'valid')

    # =========================================================
    # 5. 测试
    #    测试集 batch 同样按 5 个值解包
    # =========================================================
    def test(self, dataModule):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.test_data_loader
        preds, reals = [], []

        ctx = (
            torch.autocast(device_type=self.device_type, dtype=torch.float16)
            if self.config.use_amp else contextlib.nullcontext()
        )

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids, route_labels = batch

                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)
                sample_ids = sample_ids.to(self.config.device).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1
                ).float().to(self.config.device)

                pred, _ = self.forward(
                    x,
                    x_mark,
                    dec_input=dec_input,
                    sample_ids=sample_ids,
                    route_labels=None,
                    mode='test'
                )

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(
            preds, reals, dataModule.mean, dataModule.std
        )

        return ErrorMetrics(reals[:, :, 2], pred_raw, self.config, 'test')