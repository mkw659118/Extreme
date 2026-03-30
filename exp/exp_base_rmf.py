
import os
import time
import copy
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

        # gate 阶段 scaler 单独放一个，避免与 backbone 阶段混用造成状态干扰
        self.gate_scaler = torch.amp.GradScaler(config.device)

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
            patience=max(1, config.patience // 5),
            threshold=1e-3
        )

    def setup_gate_optimizer(self, config):
        gate_lr = getattr(config, "gate_lr", config.lr)
        gate_decay = getattr(config, "gate_decay", config.decay)

        self.gate_optimizer = get_optimizer(
            self.model.beta_gate.parameters(),
            lr=gate_lr,
            decay=gate_decay,
            config=config
        )

        self.gate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gate_optimizer,
            mode='min',
            factor=0.5,
            patience=max(1, config.patience // 5),
            threshold=1e-3
        )

    def compute_weighted_point_loss(
        self,
        pred_raw: torch.Tensor,      # [B, pred_len]
        real_raw: torch.Tensor,      # [B, pred_len]
        route_labels: torch.Tensor,  # [B]
        normal_weight: float = 1.0,
        hard_weight: float = 1.5,
        loss_type: str = "mae",
    ) -> torch.Tensor:
        if loss_type == "mae":
            per_point_loss = torch.abs(pred_raw - real_raw)
        elif loss_type == "mse":
            per_point_loss = (pred_raw - real_raw) ** 2
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        per_sample_loss = per_point_loss.mean(dim=1)

        sample_weights = torch.where(
            route_labels == 1,
            torch.full_like(route_labels, hard_weight, dtype=pred_raw.dtype),
            torch.full_like(route_labels, normal_weight, dtype=pred_raw.dtype),
        ).to(pred_raw.device)

        weighted_loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        return weighted_loss

    def RunOnce(self, config, runId, model, datamodule, log):
        try:
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

        gate_epochs = getattr(config, "gate_epochs", 5)

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

                # 先构建 retrieval index
                train_data = datamodule.train_data_loader.dataset
                train_loader = datamodule.train_data_loader
                model.prepare_retrieval_index(train_data, train_loader)

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
            # =========================
            # Stage 1: 训练 backbone
            # =========================
            model.setup_optimizer(config)
            train_time = []

            for epoch in trange(config.epochs, desc="Backbone Training"):
                model.current_epoch = epoch + 1

                if monitor.early_stop:
                    break

                train_loss, time_cost = model.train_one_epoch(datamodule, stage='backbone')
                train_time.append(time_cost)

                valid_error = model.valid(datamodule, stage='backbone')

                model.scheduler.step(valid_error[config.monitor_metric])
                print(f"[Backbone] Current LR: {model.optimizer.param_groups[0]['lr']:.6g}")

                monitor.track_one_epoch(epoch, model, valid_error, config.monitor_metric)

                log.show_epoch_error(runId, epoch, monitor, train_loss, valid_error, train_time)
                log.plotter.append_epochs(train_loss, valid_error)

                torch.save(model.state_dict(), model_path)

            # 载入最优 backbone
            model.load_state_dict(monitor.best_model)

            # 构建 retrieval index
            train_data = datamodule.train_data_loader.dataset
            train_loader = datamodule.train_data_loader
            model.prepare_retrieval_index(train_data, train_loader)

            # =========================
            # Stage 2: 冻结 backbone，只训练 gate
            # =========================
            gate_time = []
            if gate_epochs > 0:
                model.model.freeze_backbone_for_gate()
                model.setup_gate_optimizer(config)

                best_gate_metric = float("inf")
                best_gate_state = copy.deepcopy(model.state_dict())
                gate_patience = getattr(config, "gate_patience", config.patience)
                gate_wait = 0

                for gate_epoch in trange(gate_epochs, desc="Gate Training"):
                    model.current_epoch = gate_epoch + 1

                    train_loss_gate, time_cost_gate = model.train_one_epoch(datamodule, stage='gate')
                    gate_time.append(time_cost_gate)

                    valid_error_gate = model.valid(datamodule, stage='gate')

                    gate_metric = valid_error_gate[config.monitor_metric]
                    model.gate_scheduler.step(gate_metric)

                    print(f"[Gate] Current LR: {model.gate_optimizer.param_groups[0]['lr']:.6g}")

                    if gate_metric < best_gate_metric:
                        best_gate_metric = gate_metric
                        best_gate_state = copy.deepcopy(model.state_dict())
                        gate_wait = 0
                    else:
                        gate_wait += 1
                        if gate_wait >= gate_patience:
                            print("[Gate] Early stopping")
                            break

                model.load_state_dict(best_gate_state)

            # gate 训练完成后打标记，测试时启用 learned beta
            model.model.mark_gate_ready(True)
            model.model.unfreeze_all()

            sum_time = sum(train_time[: monitor.best_epoch]) + sum(gate_time)

            results = model.test(datamodule)
            log.show_test_error(runId, monitor, results, sum_time)

            torch.save(model.state_dict(), model_path)
            log(f'Model parameters saved to {model_path}')

        results['train_time'] = sum_time
        return results

    # =========================================================
    # 1. 将“预测的差分域”还原回“原始值域”
    # =========================================================
    def _restore_to_raw(self, pred, label, mean, std):
        mean_t = torch.as_tensor(mean, device=pred.device, dtype=pred.dtype)
        std_t = torch.as_tensor(std, device=pred.device, dtype=pred.dtype)

        pred_diff_z = pred[:, :, 0]
        pred_diff = pred_diff_z * std_t + mean_t

        y_pre = label[:, 0, 1]
        pred_raw = y_pre.unsqueeze(1) + torch.cumsum(pred_diff, dim=1)
        real_raw = label[:, :, 2]

        return pred_raw, real_raw

    # =========================================================
    # 2. 构建 retrieval index
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

                    self.model.add_key_value(
                        batch_x,
                        batch_y[:, -self.config.pred_len:, :],
                        sample_ids
                    )

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1}".format(i + 1, epoch + 1))
                        speed = (time.time() - time_now) / max(iter_count, 1)
                        left_time = speed * ((1 - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

        print('*******Finishing the Retrieval Indexes*********')
        self.model.value_permute = self.model.values.permute(2, 0, 1)

    # =========================================================
    # 3. 训练一个 epoch
    #    stage='backbone' -> 原始训练
    #    stage='gate'     -> 冻结 backbone，只训 beta_gate
    # =========================================================
    def train_one_epoch(self, dataModule, stage='backbone'):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()

        total_loss = 0.0
        sample_count = 0

        scaler = self.scaler if stage == 'backbone' else self.gate_scaler
        optimizer = self.optimizer if stage == 'backbone' else self.gate_optimizer
        mode = 'train' if stage == 'backbone' else 'gate_train'

        try:
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids, route_labels = train_batch

                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()
                route_labels = route_labels.to(self.config.device, non_blocking=True).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1
                ).float().to(self.config.device)

                optimizer.zero_grad(set_to_none=True)

                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred, aux_loss = self.forward(
                            x,
                            x_mark,
                            dec_input=dec_input,
                            sample_ids=sample_ids,
                            route_labels=route_labels,
                            mode=mode
                        )

                        pred_raw, real_raw = self._restore_to_raw(
                            pred, label, dataModule.mean, dataModule.std
                        )

                        if stage == 'backbone':
                            main_loss = compute_loss(
                                self,
                                x,
                                pred_raw.float(),
                                real_raw.float(),
                                self.config
                            )
                        else:
                            main_loss = self.compute_weighted_point_loss(
                                pred_raw=pred_raw,
                                real_raw=real_raw,
                                route_labels=route_labels,
                                normal_weight=1.0,
                                hard_weight=1.5,
                                loss_type="mae",
                            )

                        total_loss_value = main_loss + aux_loss

                    scaler.scale(total_loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    pred, aux_loss = self.forward(
                        x,
                        x_mark,
                        dec_input=dec_input,
                        sample_ids=sample_ids,
                        route_labels=route_labels,
                        mode=mode
                    )

                    pred_raw, real_raw = self._restore_to_raw(
                        pred, label, dataModule.mean, dataModule.std
                    )

                    if stage == 'backbone':
                        main_loss = self.compute_weighted_point_loss(
                            pred_raw=pred_raw,
                            real_raw=real_raw,
                            route_labels=route_labels,
                            normal_weight=1.0,
                            hard_weight=1.5,
                            loss_type="mae",
                        )
                    else:
                        main_loss = self.compute_weighted_point_loss(
                            pred_raw=pred_raw,
                            real_raw=real_raw,
                            route_labels=route_labels,
                            normal_weight=1.0,
                            hard_weight=1.5,
                            loss_type="mae",
                        )

                    total_loss_value = main_loss + aux_loss

                    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataModule.train_data_loader):
                        beta_mean_item = 0.0
                        if hasattr(self.model, "latest_aux_dict") and "beta_mean" in self.model.latest_aux_dict:
                            beta_mean_item = float(self.model.latest_aux_dict["beta_mean"].item())

                        print(
                            f"[{stage}] Batch {batch_idx+1}/{len(dataModule.train_data_loader)} - "
                            f"Main Loss: {main_loss.item():.6f}, "
                            f"Aux Loss: {aux_loss.item():.6f}, "
                            f"Beta Mean: {beta_mean_item:.6f}, "
                            f"Total Loss: {total_loss_value.item():.6f}"
                        )

                    total_loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += total_loss_value.item() * x.size(0)
                sample_count += x.size(0)

        except Exception as e:
            print(f"[Train Error] Stage={stage}, Epoch {self.current_epoch} failed: {str(e)}")
            raise

        finally:
            self.eval()
            torch.set_grad_enabled(False)
            t2 = time.time()

            avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
            print(
                f"[Train-{stage}] Epoch {self.current_epoch} finished | "
                f"Avg Loss: {avg_loss:.6f} | Time Cost: {t2 - t1:.2f}s"
            )

        return avg_loss, t2 - t1

    # =========================================================
    # 4. 验证
    # =========================================================
    def valid(self, dataModule, stage='backbone'):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader
        preds, reals = [], []

        mode = 'valid' if stage == 'backbone' else 'gate_valid'

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
                    mode=mode
                )

                reals.append(label)
                preds.append(pred)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        pred_raw, real_raw = self._restore_to_raw(
            preds, reals, dataModule.mean, dataModule.std
        )

        return ErrorMetrics(reals[:, :, 2], pred_raw, self.config, 'valid')

    # =========================================================
    # 5. 测试
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
