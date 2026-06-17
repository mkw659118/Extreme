import contextlib
import copy
import os
import pickle
import time

import numpy as np
import torch
from tqdm import trange

from exp.exp_loss import compute_loss
from exp.exp_metrics_mask import ErrorMetrics
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

    def _prepare_label_mask(self, x_mark, label):
        future_label = label[:, -self.pred_len:, :]
        mask = None

        if x_mark is not None and torch.is_tensor(x_mark):
            candidate = x_mark
            if candidate.dim() == 2:
                candidate = candidate.unsqueeze(-1)
            if candidate.dim() == 3 and candidate.shape[1] >= self.pred_len:
                mask = candidate[:, -self.pred_len:, :]
                if mask.shape[-1] != future_label.shape[-1]:
                    mask = mask[..., : future_label.shape[-1]]
                if mask.shape[:2] != future_label.shape[:2]:
                    mask = None

        if mask is None:
            mask = torch.isfinite(future_label).float()
            if bool(getattr(self.config, "mask_zero_as_missing", True)):
                mask = mask * (future_label != 0).float()

        return mask.to(device=label.device, dtype=label.dtype)

    def _time_feature_dim(self):
        freq = str(getattr(self.config, "freq", "h"))
        freq_map = {
            "h": 4,
            "t": 5,
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }
        return int(freq_map.get(freq, 4))

    def _prepare_model_x_mark(self, x, x_mark):
        """Keep label masks out of models that expect time-feature marks."""
        expected_dim = self._time_feature_dim()

        if x_mark is not None and torch.is_tensor(x_mark):
            candidate = x_mark
            if candidate.dim() == 2:
                candidate = candidate.unsqueeze(-1)
            if (
                candidate.dim() == 3
                and candidate.shape[1] == x.shape[1]
                and candidate.shape[-1] == expected_dim
            ):
                return candidate.to(device=x.device, dtype=x.dtype)

        return torch.zeros(
            x.shape[0],
            x.shape[1],
            expected_dim,
            device=x.device,
            dtype=x.dtype,
        )

    def _compute_masked_supervised_loss(self, x, pred_scaled, real_scaled, label_mask):
        valid = (
            (label_mask > 0.5)
            & torch.isfinite(pred_scaled)
            & torch.isfinite(real_scaled)
        )
        if valid.shape != pred_scaled.shape:
            valid = valid.expand_as(pred_scaled)

        if not torch.any(valid):
            return pred_scaled.sum() * 0.0

        pred_valid = pred_scaled[valid].reshape(1, -1, 1)
        real_valid = real_scaled[valid].reshape(1, -1, 1)

        return compute_loss(
            self,
            x,
            pred_valid.float(),
            real_valid.float(),
            self.config,
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
                label_mask = self._prepare_label_mask(
                    x_mark,
                    label,
                )
                model_x_mark = self._prepare_model_x_mark(
                    x,
                    x_mark,
                )

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
                            model_x_mark,
                            dec_input,
                            None,
                        )

                        pred_scaled = pred[:, -self.pred_len:, 0:1]
                        real_scaled = label[:, -self.pred_len:, 0:1]

                        main_loss = self._compute_masked_supervised_loss(
                            x,
                            pred_scaled,
                            real_scaled,
                            label_mask,
                        )

                        total_loss_value = main_loss

                    scaler.scale(total_loss_value).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    pred = self.forward(
                        x,
                        model_x_mark,
                        dec_input,
                        None,
                    )

                    pred_scaled = pred[:, -self.pred_len:, 0:1]
                    real_scaled = label[:, -self.pred_len:, 0:1]

                    main_loss = self._compute_masked_supervised_loss(
                        x,
                        pred_scaled,
                        real_scaled,
                        label_mask,
                    )

                    total_loss_value = main_loss
                    # print(
                    #     f"[{stage}] Batch {batch_idx + 1}/{len(dataModule.train_data_loader)} - "
                    #     f"Main Loss: {main_loss.item():.6f}, "
                    #     f"Total Loss: {total_loss_value.item():.6f}"
                    # )

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

    def _inverse_to_original_space(self, dataModule, value_scaled: torch.Tensor) -> torch.Tensor:
        value_np = value_scaled.detach().float().cpu().numpy()
        value_original = dataModule.inverse_value_norm(value_np).astype(np.float32)
        return torch.from_numpy(value_original).to(self.config.device)

    def _evaluate_original_space(self, dataModule, dataloader, mode: str):
        
        preds = []
        reals = []
        masks = []

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
                label_mask = self._prepare_label_mask(
                    x_mark,
                    label,
                )
                model_x_mark = self._prepare_model_x_mark(
                    x,
                    x_mark,
                )

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                pred = self.forward(
                    x,
                    model_x_mark,
                    dec_input,
                    None,
                )

                pred_scaled = pred[:, -self.pred_len:, 0:1]
                real_scaled = label[:, -self.pred_len:, 0:1]

                pred_original = self._inverse_to_original_space(dataModule, pred_scaled)
                real_original = self._inverse_to_original_space(dataModule, real_scaled)

                preds.append(pred_original)
                reals.append(real_original)
                masks.append(label_mask)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        masks = torch.cat(masks, dim=0)

        return ErrorMetrics(
            reals,
            preds,
            self.config,
            mode,
            valid_mask=masks,
        )

    def valid(self, dataModule, stage="backbone"):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader
        mode = "valid" if stage == "backbone" else "gate_valid"

        return self._evaluate_original_space(dataModule, dataloader, mode)

    def _build_test_save_paths(self):
        dataset_name = getattr(self.config, "dataset", "Dataset")
        model_name = getattr(self.config, "model", "Model")

        data_file = getattr(self.config, "data_file", "data")
        data_tag = os.path.splitext(os.path.basename(str(data_file)))[0]

        target_col = int(getattr(self.config, "target_col", 0))
        original_target_col = int(getattr(self.config, "original_target_col", target_col))

        save_dir = os.path.join(
            "./draw",
            str(dataset_name),
            str(data_tag),
            str(model_name),
            f"PL{self.config.pred_len}_DM{self.config.d_model}",
            f"TC{original_target_col}",
        )

        return (
            os.path.join(save_dir, "test_raw.csv"),
            os.path.join(save_dir, "test_agg.csv"),
        )

    @staticmethod
    def _to_numpy_array(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _save_test_raw_result(self, true_series, pred_series, save_path):
        true_series = self._to_numpy_array(true_series)
        pred_series = self._to_numpy_array(pred_series)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true and pred shapes do not match: {true_series.shape} vs {pred_series.shape}"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if true_series.ndim <= 2 or true_series.shape[-1] == 1:
            true_flat = true_series.reshape(-1)
            pred_flat = pred_series.reshape(-1)
            rows = np.column_stack([np.arange(len(true_flat)), true_flat, pred_flat])
            np.savetxt(
                save_path,
                rows,
                delimiter=",",
                header="index,true,pred",
                comments="",
                fmt=["%d", "%.10g", "%.10g"],
            )
        else:
            true_flat = true_series.reshape(-1, true_series.shape[-1])
            pred_flat = pred_series.reshape(-1, pred_series.shape[-1])
            rows = np.concatenate([true_flat, pred_flat], axis=1)
            header = ",".join(
                [f"true_var_{i}" for i in range(true_flat.shape[1])]
                + [f"pred_var_{i}" for i in range(pred_flat.shape[1])]
            )
            np.savetxt(save_path, rows, delimiter=",", header=header, comments="", fmt="%.10g")

        print(f"[Save] test raw result saved to: {save_path}")

    def _save_test_aggregated_result(self, true_series, pred_series, sample_ids, save_path):
        true_series = self._to_numpy_array(true_series)
        pred_series = self._to_numpy_array(pred_series)
        sample_ids = self._to_numpy_array(sample_ids).astype(np.int64)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true and pred shapes do not match: {true_series.shape} vs {pred_series.shape}"
            )

        if true_series.ndim == 2:
            true_series = true_series[:, :, None]
            pred_series = pred_series[:, :, None]

        if true_series.ndim != 3:
            raise ValueError(f"Expected [N, pred_len, C], got shape {true_series.shape}")

        if len(sample_ids) != true_series.shape[0]:
            raise ValueError(
                f"sample_ids length {len(sample_ids)} does not match window count {true_series.shape[0]}"
            )

        pred_len = true_series.shape[1]
        out_dim = true_series.shape[2]
        seq_len = int(self.config.seq_len)

        true_sum = {}
        pred_sum = {}
        count = {}

        for window_idx, start in enumerate(sample_ids):
            base_time_idx = int(start) + seq_len

            for horizon in range(pred_len):
                time_idx = base_time_idx + horizon

                if time_idx not in count:
                    true_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    pred_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    count[time_idx] = 0

                true_sum[time_idx] += true_series[window_idx, horizon].astype(np.float64)
                pred_sum[time_idx] += pred_series[window_idx, horizon].astype(np.float64)
                count[time_idx] += 1

        time_indices = sorted(count)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if out_dim == 1:
            rows = np.asarray(
                [
                    [
                        time_idx,
                        count[time_idx],
                        true_sum[time_idx][0] / count[time_idx],
                        pred_sum[time_idx][0] / count[time_idx],
                    ]
                    for time_idx in time_indices
                ],
                dtype=np.float64,
            )
            np.savetxt(
                save_path,
                rows,
                delimiter=",",
                header="time_idx,count,true,pred",
                comments="",
                fmt=["%d", "%d", "%.10g", "%.10g"],
            )
        else:
            rows = []
            for time_idx in time_indices:
                true_avg = true_sum[time_idx] / count[time_idx]
                pred_avg = pred_sum[time_idx] / count[time_idx]
                rows.append(np.concatenate([[time_idx, count[time_idx]], true_avg, pred_avg]))

            rows = np.asarray(rows, dtype=np.float64)
            header = ",".join(
                ["time_idx", "count"]
                + [f"true_var_{i}" for i in range(out_dim)]
                + [f"pred_var_{i}" for i in range(out_dim)]
            )
            fmt = ["%d", "%d"] + ["%.10g"] * (2 * out_dim)
            np.savetxt(save_path, rows, delimiter=",", header=header, comments="", fmt=fmt)

        print(f"[Save] test aggregated result saved to: {save_path}")
    
    def _aggregate_series_by_time_for_metric(
        self,
        true_series,
        pred_series,
        sample_ids,
        valid_mask=None,
    ):
        true_series = self._to_numpy_array(true_series)
        pred_series = self._to_numpy_array(pred_series)
        sample_ids = self._to_numpy_array(sample_ids).astype(np.int64)
        if valid_mask is not None:
            valid_mask = self._to_numpy_array(valid_mask)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true and pred shapes do not match: {true_series.shape} vs {pred_series.shape}"
            )

        if true_series.ndim == 2:
            true_series = true_series[:, :, None]
            pred_series = pred_series[:, :, None]

        if true_series.ndim != 3:
            raise ValueError(f"Expected [N, pred_len, C], got shape {true_series.shape}")

        if valid_mask is None:
            valid_mask = np.ones_like(true_series, dtype=np.float32)
        else:
            if valid_mask.ndim == 2:
                valid_mask = valid_mask[:, :, None]
            if valid_mask.shape != true_series.shape:
                if valid_mask.shape[-1] == 1 and true_series.shape[-1] > 1:
                    valid_mask = np.broadcast_to(valid_mask, true_series.shape)
                else:
                    raise ValueError(
                        f"valid_mask shape {valid_mask.shape} does not match "
                        f"series shape {true_series.shape}"
                    )

        pred_len = true_series.shape[1]
        out_dim = true_series.shape[2]
        seq_len = int(self.config.seq_len)

        true_sum = {}
        pred_sum = {}
        count = {}

        for window_idx, start in enumerate(sample_ids):
            base_time_idx = int(start) + seq_len

            for horizon in range(pred_len):
                time_idx = base_time_idx + horizon

                point_mask = (
                    (valid_mask[window_idx, horizon] > 0.5)
                    & np.isfinite(true_series[window_idx, horizon])
                    & np.isfinite(pred_series[window_idx, horizon])
                )
                if not np.any(point_mask):
                    continue

                if time_idx not in count:
                    true_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    pred_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    count[time_idx] = np.zeros(out_dim, dtype=np.float64)

                true_sum[time_idx][point_mask] += true_series[
                    window_idx,
                    horizon,
                ][point_mask].astype(np.float64)
                pred_sum[time_idx][point_mask] += pred_series[
                    window_idx,
                    horizon,
                ][point_mask].astype(np.float64)
                count[time_idx][point_mask] += 1.0

        time_indices = [i for i in sorted(count) if np.all(count[i] > 0)]
        if not time_indices:
            raise ValueError("No valid entries for aggregated metric computation.")

        true_agg = np.stack(
            [true_sum[i] / count[i] for i in time_indices],
            axis=0,
        ).astype(np.float32)

        pred_agg = np.stack(
            [pred_sum[i] / count[i] for i in time_indices],
            axis=0,
        ).astype(np.float32)

        return true_agg, pred_agg

    def test(self, dataModule):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.test_data_loader

        preds = []
        reals = []
        ids = []
        masks = []

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
                label_mask = self._prepare_label_mask(
                    x_mark,
                    label,
                )
                model_x_mark = self._prepare_model_x_mark(
                    x,
                    x_mark,
                )

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                pred = self.forward(
                    x,
                    model_x_mark,
                    dec_input,
                    None,
                )

                pred_scaled = pred[:, -self.pred_len:, 0:1]
                real_scaled = label[:, -self.pred_len:, 0:1]

                pred_original = self._inverse_to_original_space(dataModule, pred_scaled)
                real_original = self._inverse_to_original_space(dataModule, real_scaled)

                preds.append(pred_original)
                reals.append(real_original)
                ids.append(sample_ids)
                masks.append(label_mask)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        ids = torch.cat(ids, dim=0)
        masks = torch.cat(masks, dim=0)

        save_path, aggregated_save_path = self._build_test_save_paths()
        self._save_test_raw_result(
            true_series=reals,
            pred_series=preds,
            save_path=save_path,
        )
        self._save_test_aggregated_result(
            true_series=reals,
            pred_series=preds,
            sample_ids=ids,
            save_path=aggregated_save_path,
        )

        real_agg, pred_agg = self._aggregate_series_by_time_for_metric(
            true_series=reals,
            pred_series=preds,
            sample_ids=ids,
            valid_mask=masks,
        )

        return ErrorMetrics(
            real_agg,
            pred_agg,
            self.config,
            "test",
        )

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
