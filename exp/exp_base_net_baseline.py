import contextlib
import copy
import os
import pickle
import time

import numpy as np
import torch
from tqdm import trange

from exp.exp_loss import compute_loss
from exp.exp_metrics import ErrorMetrics
from utils.model_monitor import EarlyStopping
from utils.model_trainer import get_loss_function, get_optimizer


class BasicModel(torch.nn.Module):
    def __init__(self, config):
        super(BasicModel, self).__init__()

        self.config = config
        self.pred_len = config.pred_len
        self.label_len = config.label_len

        device_str = str(config.device)
        self.device_type = "cuda" if "cuda" in device_str else "cpu"

        self.scaler = torch.amp.GradScaler(config.device)
        self.current_epoch = 0
        self.gate_scaler = torch.amp.GradScaler(config.device)
        self.pretrain_scaler = torch.amp.GradScaler(config.device)

    def forward(self, *x, **kwargs):
        return self.model(*x, **kwargs)

    def setup_optimizer(self, config):
        self.to(config.device)
        self.loss_function = get_loss_function(config).to(config.device)

        self.optimizer = get_optimizer(
            self.parameters(),
            lr=config.lr,
            decay=config.decay,
            config=config,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, config.patience // 5),
            threshold=1e-3,
        )

    def train_one_epoch(self, dataModule, stage="backbone"):
        self.train()
        torch.set_grad_enabled(True)

        t1 = time.time()

        total_loss = 0.0
        sample_count = 0

        scaler = self.scaler if stage == "backbone" else self.gate_scaler
        optimizer = self.optimizer if stage == "backbone" else self.gate_optimizer
        mode = "train" if stage == "backbone" else "gate_train"

        try:
            for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
                x, x_mark, label, sample_ids = train_batch

                x = x.to(self.config.device, non_blocking=True)
                x_mark = x_mark.to(self.config.device, non_blocking=True)
                label = label.to(self.config.device, non_blocking=True)
                sample_ids = sample_ids.to(self.config.device, non_blocking=True).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                optimizer.zero_grad(set_to_none=True)

                if self.config.use_amp:
                    with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                        pred = self.forward(
                            x,
                            x_mark,
                            dec_input,
                            None,
                        )

                        main_loss = compute_loss(
                            self,
                            pred,
                            label,
                            self.config,
                        )
                        print(pred.shape, label.shape)

                        total_loss_value = main_loss

                    scaler.scale(total_loss_value).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    pred = self.forward(
                        x,
                        x_mark,
                        dec_input,
                        None,
                    )

                    main_loss = compute_loss(
                        self,
                        x,
                        pred,
                        label,
                        self.config,
                    )

                    total_loss_value = main_loss
                    print(
                        f"[{stage}] Batch {batch_idx + 1}/{len(dataModule.train_data_loader)} - "
                        f"Main Loss: {main_loss.item():.6f}, "
                        f"Total Loss: {total_loss_value.item():.6f}"
                    )

                    total_loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    optimizer.step()

                total_loss += total_loss_value.item() * x.size(0)
                sample_count += x.size(0)

        except Exception as e:
            print(
                f"[Train Error] Stage={stage}, Epoch {self.current_epoch} failed: {str(e)}"
            )
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

    def _metric_config_for_original_space(self):
        """
        Metrics are computed after inverse normalization, so ErrorMetrics should
        treat the input tensors as already being in the original raw-value space.
        """
        metric_config = copy.copy(self.config)
        setattr(metric_config, "eval_space", "original")
        setattr(metric_config, "metric_space", "original")
        setattr(metric_config, "already_original_space", True)

        # If ErrorMetrics has optional inverse/recover switches, disable them
        # here to avoid applying inverse normalization twice.
        for flag in (
            "inverse_eval",
            "denormalize_eval",
            "recover_level",
            "recover_level_from_diff",
            "use_inverse_transform",
            "use_denormalize",
        ):
            if hasattr(metric_config, flag):
                setattr(metric_config, flag, False)

        return metric_config

    def _inverse_to_original_space(self, dataModule, value_scaled: torch.Tensor) -> torch.Tensor:
        """
        Convert standardized prediction/label values back to raw-value space.

        The dataset normalizer is fitted only on the training split, so calling
        dataModule.inverse_value_norm here keeps validation/test evaluation on
        the original scale without leaking validation/test statistics.
        """
        if not hasattr(dataModule, "inverse_value_norm"):
            raise AttributeError(
                "dataModule must provide inverse_value_norm() for original-space evaluation."
            )

        value_np = value_scaled.detach().float().cpu().numpy()
        value_original = dataModule.inverse_value_norm(value_np).astype(np.float32)
        return torch.from_numpy(value_original).to(self.config.device)

    def _evaluate_original_space(self, dataModule, dataloader, mode: str):
        """
        Validation/test metric computation.

        Model outputs and labels are standardized tensors. Before computing
        MAE/RMSE/MAPE/R2 and other reported metrics, both are inverse-normalized
        to the original raw-value space.
        """
        preds = []
        reals = []

        ctx = (
            torch.autocast(device_type=self.device_type, dtype=torch.float16)
            if self.config.use_amp
            else contextlib.nullcontext()
        )

        with ctx:
            for batch in dataloader:
                x, x_mark, label, sample_ids = batch

                x = x.to(self.config.device)
                x_mark = x_mark.to(self.config.device)
                label = label.to(self.config.device)

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                pred = self.forward(
                    x,
                    x_mark,
                    dec_input,
                    None,
                )

                pred_scaled = pred[:, -self.pred_len:, 0:1]
                real_scaled = label[:, -self.pred_len:, 0:1]

                pred_original = self._inverse_to_original_space(dataModule, pred_scaled)
                real_original = self._inverse_to_original_space(dataModule, real_scaled)

                preds.append(pred_original)
                reals.append(real_original)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)

        return ErrorMetrics(
            reals,
            preds,
            self._metric_config_for_original_space(),
            mode,
        )

    def valid(self, dataModule, stage="backbone"):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader
        mode = "valid" if stage == "backbone" else "gate_valid"

        return self._evaluate_original_space(dataModule, dataloader, mode)

    def test(self, dataModule):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.test_data_loader
        return self._evaluate_original_space(dataModule, dataloader, "test")

    def _need_retrain(self, config, runId, log):
        model_path = f"./checkpoints/{config.model}/{log.filename}_round_{runId}.pt"

        return (
            config.retrain == 1
            or (not os.path.exists(model_path) and config.continue_train)
        )

    def _run_backbone_gate_once(self, config, runId, model, datamodule, log):
        try:
            model.compile()
        except Exception as e:
            print(f"Skip the model.compile() because {e}")

        monitor = EarlyStopping(config)

        os.makedirs(f"./checkpoints/{config.model}", exist_ok=True)

        model_path = f"./checkpoints/{config.model}/{log.filename}_round_{runId}.pt"

        retrain_required = (
            config.retrain == 1
            or (not os.path.exists(model_path) and config.continue_train)
        )

        if not retrain_required:
            try:
                sum_time = pickle.load(
                    open(f"./results/metrics/" + log.filename + ".pkl", "rb")
                )["train_time"][runId]

                model.load_state_dict(
                    torch.load(model_path, weights_only=True, map_location="cpu")
                )

                model.setup_optimizer(config)

                results = model.test(datamodule)

                log.show_results(results, sum_time)

                config.record = False

            except Exception as e:
                log.only_print(f"Error: {str(e)}")
                retrain_required = True

        if config.continue_train:
            log.only_print("Continue training...")

            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location="cpu")
            )

        if retrain_required:
            model.setup_optimizer(config)

            train_time = []

            for epoch in trange(config.epochs, desc="Backbone Training"):
                model.current_epoch = epoch + 1

                if monitor.early_stop:
                    break

                train_loss, time_cost = model.train_one_epoch(
                    datamodule,
                    stage="backbone",
                )

                train_time.append(time_cost)

                valid_error = model.valid(datamodule, stage="backbone")

                model.scheduler.step(valid_error[config.monitor_metric])

                print(
                    f"[Backbone] Current LR: {model.optimizer.param_groups[0]['lr']:.6g}"
                )

                monitor.track_one_epoch(
                    epoch,
                    model,
                    valid_error,
                    config.monitor_metric,
                )

                log.show_epoch_error(
                    runId,
                    epoch,
                    monitor,
                    train_loss,
                    valid_error,
                    train_time,
                )

                log.plotter.append_epochs(train_loss, valid_error)

                torch.save(model.state_dict(), model_path)

            model.load_state_dict(monitor.best_model)

            
            sum_time = sum(train_time[: monitor.best_epoch]) 
            results = model.test(datamodule)

            log.show_test_error(runId, monitor, results, sum_time)

            torch.save(model.state_dict(), model_path)

            log(f"Model parameters saved to {model_path}")

        results["train_time"] = sum_time

        return results

    def RunOnce(self, config, runId, model, datamodule, log):
        return self._run_backbone_gate_once(
            config,
            runId,
            model,
            datamodule,
            log,
        )
