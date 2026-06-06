import contextlib
import copy
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from exp.exp_loss import compute_loss
from exp.exp_metrics_mask import ErrorMetrics
from utils.model_monitor import EarlyStopping
from utils.model_trainer import get_loss_function, get_optimizer
from torch.utils.data import DataLoader


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

    def setup_gate_optimizer(self, config):
        gate_lr = getattr(config, "gate_lr", config.lr)
        gate_decay = getattr(config, "gate_decay", config.decay)

        self.gate_optimizer = get_optimizer(
            self.model.beta_gate.parameters(),
            lr=gate_lr,
            decay=gate_decay,
            config=config,
        )

        self.gate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gate_optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, config.patience // 5),
            threshold=1e-3,
        )

    def setup_pretrain_optimizer(self, config):
        self.to(config.device)

        pretrain_lr = getattr(config, "pretrain_lr", config.lr)
        pretrain_decay = getattr(config, "pretrain_decay", config.decay)

        self.pretrain_optimizer = torch.optim.AdamW(
            self.model.get_state_prior_parameters(),
            lr=pretrain_lr,
            weight_decay=pretrain_decay,
        )

        self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pretrain_optimizer,
            mode="min",
            factor=0.5,
            patience=max(1, config.patience // 5),
            threshold=1e-4,
        )

    def _select_target_if_needed(self, values, dataModule):
        target_dim = int(
            getattr(dataModule, "target_dim", getattr(self.config, "target_dim", 0))
        )

        if target_dim <= 0 or values.shape[-1] == target_dim:
            return values

        target_col = int(
            getattr(dataModule, "target_col", getattr(self.config, "target_col", 0))
        )
        end_col = target_col + target_dim

        if end_col > values.shape[-1]:
            raise ValueError(
                f"Cannot select target columns [{target_col}:{end_col}] from "
                f"values with shape {tuple(values.shape)}."
            )

        return values[..., target_col:end_col]

    def _prepare_scaled_prediction_and_label(self, pred, label, dataModule):
        """
        Align model prediction and label in the standardized space.

        Training loss should use the returned tensors directly. Validation and
        testing restore these tensors to the original raw-value space before
        computing metrics.
        """
        pred = pred[:, -self.pred_len:, :]
        future_label = label[:, -self.pred_len:, :]

        pred = self._select_target_if_needed(pred, dataModule)
        future_label = self._select_target_if_needed(future_label, dataModule)

        if pred.shape[-1] != future_label.shape[-1]:
            raise ValueError(
                f"Prediction and label target dimensions mismatch: "
                f"{tuple(pred.shape)} vs {tuple(future_label.shape)}."
            )

        return pred, future_label

    def _prepare_label_mask(self, x_mark, label, dataModule):
        """
        Extract future-label valid mask from x_mark.

        In the missing-aware dataset, x_mark is reused as y_mask with shape:
            [B, pred_len, target_dim]

        The mask indicates whether each future first-order-difference target is
        valid. Positions with mask=0 do not participate in supervised loss.

        If an old dataset is used and x_mark does not match this convention, we
        fall back to a finite-value mask. If config.mask_zero_as_missing=True,
        zero labels are also treated as invalid in this fallback path.
        """
        future_label = label[:, -self.pred_len:, :]

        mask = None
        if x_mark is not None and torch.is_tensor(x_mark):
            candidate = x_mark
            if candidate.dim() == 2:
                candidate = candidate.unsqueeze(-1)

            if candidate.dim() == 3 and candidate.shape[1] >= self.pred_len:
                candidate = candidate[:, -self.pred_len:, :]
                try:
                    mask = self._select_target_if_needed(candidate, dataModule)
                except Exception:
                    mask = candidate

                if mask.shape[:2] != future_label.shape[:2]:
                    mask = None

        if mask is None:
            mask = torch.isfinite(future_label).float()
            if bool(getattr(self.config, "mask_zero_as_missing", True)):
                mask = mask * (future_label != 0).float()

        target_dim = int(
            getattr(dataModule, "target_dim", getattr(self.config, "target_dim", 0))
        )
        if target_dim > 0 and mask.shape[-1] != target_dim:
            target_col = int(
                getattr(dataModule, "target_col", getattr(self.config, "target_col", 0))
            )
            if mask.shape[-1] > target_col:
                mask = mask[..., target_col : target_col + target_dim]
            else:
                mask = mask[..., :target_dim]

        return mask.to(device=label.device, dtype=label.dtype)

    def _compute_masked_supervised_loss(self, x, pred_scaled, real_scaled, label_mask):
        """
        Keep the original supervised objective from compute_loss(), but apply it
        only on valid future-label positions.

        This does not introduce a new training objective. It only removes
        missing target positions from the existing objective.
        """
        if label_mask is None:
            return compute_loss(
                self,
                x,
                pred_scaled.float(),
                real_scaled.float(),
                self.config,
            )

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

    @staticmethod
    def _recover_level_mask_from_diff_mask(diff_mask):
        """
        Convert future first-diff valid mask to raw-level valid mask.

        Because raw prediction at horizon h is reconstructed by cumulative sum
        of differences from 1..h, the raw value at h is valid only if all
        previous future differences used in the cumulative path are valid.
        """
        if diff_mask is None:
            return None
        return torch.cumprod((diff_mask > 0.5).float(), dim=1)


    def _get_value_mean_std(self, dataModule, out_dim, device, dtype):
        """Get mean/std for inverse z-score normalization of the target columns."""
        mean = getattr(dataModule, "mean", None)
        std = getattr(dataModule, "std", None)

        if mean is None and hasattr(dataModule, "get_mean"):
            mean = dataModule.get_mean()
        if std is None and hasattr(dataModule, "get_std"):
            std = dataModule.get_std()

        if mean is None or std is None:
            raise AttributeError(
                "dataModule must provide mean/std or get_mean()/get_std() "
                "for inverse normalization."
            )

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        target_col = int(
            getattr(dataModule, "target_col", getattr(self.config, "target_col", 0))
        )
        target_dim = int(
            getattr(dataModule, "target_dim", getattr(self.config, "target_dim", out_dim))
        )

        if mean.shape[-1] == out_dim:
            mean = mean[:out_dim]
            std = std[:out_dim]
        elif target_dim > 0 and out_dim == target_dim:
            end_col = target_col + target_dim
            mean = mean[target_col:end_col]
            std = std[target_col:end_col]
        elif out_dim == 1:
            mean = mean[target_col : target_col + 1]
            std = std[target_col : target_col + 1]
        else:
            raise ValueError(
                f"Cannot map out_dim={out_dim} to normalizer with "
                f"mean shape {mean.shape}."
            )

        mean_t = torch.as_tensor(mean, device=device, dtype=dtype).view(1, 1, -1)
        std_t = torch.as_tensor(std, device=device, dtype=dtype).view(1, 1, -1)

        return mean_t, std_t

    def _restore_to_raw(self, pred, label, dataModule, sample_ids=None):
        """
        Restore predictions/labels to the original evaluation space.

        - If the dataset provides recover_level_from_diff(), treat outputs as
          standardized first-order differences and recover raw values by
          inverse-normalizing + cumulative summation.
        - Otherwise fall back to plain inverse z-score normalization.
        """

        pred_scaled, real_scaled = self._prepare_scaled_prediction_and_label(
            pred,
            label,
            dataModule,
        )

        if sample_ids is not None and hasattr(dataModule, "recover_level_from_diff"):
            sample_ids_np = sample_ids.detach().cpu().numpy()

            pred_raw = dataModule.recover_level_from_diff(
                pred_scaled.detach().cpu().numpy(),
                sample_ids_np,
            )
            real_raw = dataModule.recover_level_from_diff(
                real_scaled.detach().cpu().numpy(),
                sample_ids_np,
            )

            return (
                torch.as_tensor(
                    pred_raw,
                    device=pred_scaled.device,
                    dtype=pred_scaled.dtype,
                ),
                torch.as_tensor(
                    real_raw,
                    device=real_scaled.device,
                    dtype=real_scaled.dtype,
                ),
            )

        mean_t, std_t = self._get_value_mean_std(
            dataModule,
            out_dim=pred_scaled.shape[-1],
            device=pred_scaled.device,
            dtype=pred_scaled.dtype,
        )

        pred_raw = pred_scaled * std_t + mean_t
        real_raw = real_scaled * std_t + mean_t

        return pred_raw, real_raw

    def save_single_model_result(self, true_series, pred_series, save_path):
        if isinstance(true_series, torch.Tensor):
            true_series = true_series.detach().cpu().numpy()
        else:
            true_series = np.asarray(true_series)

        if isinstance(pred_series, torch.Tensor):
            pred_series = pred_series.detach().cpu().numpy()
        else:
            pred_series = np.asarray(pred_series)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true 和 pred 形状不一致: {true_series.shape} vs {pred_series.shape}"
            )

        # 单变量：保存 index, true, pred
        if true_series.ndim <= 2 or true_series.shape[-1] == 1:
            true_flat = true_series.reshape(-1)
            pred_flat = pred_series.reshape(-1)

            df = pd.DataFrame(
                {
                    "index": np.arange(len(true_flat)),
                    "true": true_flat,
                    "pred": pred_flat,
                }
            )

        # 多变量：[N, pred_len, C] -> [N * pred_len, C]
        else:
            true_flat = true_series.reshape(-1, true_series.shape[-1])
            pred_flat = pred_series.reshape(-1, pred_series.shape[-1])

            result_dict = {}
            for i in range(true_flat.shape[1]):
                result_dict[f"true_var_{i}"] = true_flat[:, i]
                result_dict[f"pred_var_{i}"] = pred_flat[:, i]

            df = pd.DataFrame(result_dict)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"[Save] model result saved to: {save_path}")

    def save_aggregated_model_result(
        self,
        true_series,
        pred_series,
        sample_ids,
        save_path,
    ):
        if isinstance(true_series, torch.Tensor):
            true_series = true_series.detach().cpu().numpy()
        else:
            true_series = np.asarray(true_series)

        if isinstance(pred_series, torch.Tensor):
            pred_series = pred_series.detach().cpu().numpy()
        else:
            pred_series = np.asarray(pred_series)

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.detach().cpu().numpy()
        else:
            sample_ids = np.asarray(sample_ids)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true and pred shapes do not match: {true_series.shape} vs {pred_series.shape}"
            )

        if true_series.ndim == 2:
            true_series = true_series[:, :, None]
            pred_series = pred_series[:, :, None]

        if true_series.ndim != 3:
            raise ValueError(
                f"Expected [N, pred_len, C] series, got shape {true_series.shape}"
            )

        if len(sample_ids) != true_series.shape[0]:
            raise ValueError(
                f"sample_ids length {len(sample_ids)} does not match "
                f"window count {true_series.shape[0]}"
            )

        pred_len = true_series.shape[1]
        out_dim = true_series.shape[2]
        seq_len = int(self.config.seq_len)

        true_sum = {}
        pred_sum = {}
        count = {}

        for window_idx, start in enumerate(sample_ids.astype(np.int64)):
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
        result_dict = {
            "time_idx": time_indices,
            "count": [count[i] for i in time_indices],
        }

        if out_dim == 1:
            result_dict["true"] = [
                true_sum[i][0] / count[i] for i in time_indices
            ]
            result_dict["pred"] = [
                pred_sum[i][0] / count[i] for i in time_indices
            ]
        else:
            for dim_idx in range(out_dim):
                result_dict[f"true_var_{dim_idx}"] = [
                    true_sum[i][dim_idx] / count[i] for i in time_indices
                ]
                result_dict[f"pred_var_{dim_idx}"] = [
                    pred_sum[i][dim_idx] / count[i] for i in time_indices
                ]

        df = pd.DataFrame(result_dict)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"[Save] aggregated model result saved to: {save_path}")

    def aggregate_series_by_time(self, true_series, pred_series, sample_ids, valid_mask=None):
        if isinstance(true_series, torch.Tensor):
            true_series = true_series.detach().cpu().numpy()
        else:
            true_series = np.asarray(true_series)

        if isinstance(pred_series, torch.Tensor):
            pred_series = pred_series.detach().cpu().numpy()
        else:
            pred_series = np.asarray(pred_series)

        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.detach().cpu().numpy()
        else:
            sample_ids = np.asarray(sample_ids)

        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.detach().cpu().numpy()
            else:
                valid_mask = np.asarray(valid_mask)

        if true_series.shape != pred_series.shape:
            raise ValueError(
                f"true and pred shapes do not match: {true_series.shape} vs {pred_series.shape}"
            )

        if true_series.ndim == 2:
            true_series = true_series[:, :, None]
            pred_series = pred_series[:, :, None]

        if true_series.ndim != 3:
            raise ValueError(
                f"Expected [N, pred_len, C] series, got shape {true_series.shape}"
            )

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

        if len(sample_ids) != true_series.shape[0]:
            raise ValueError(
                f"sample_ids length {len(sample_ids)} does not match "
                f"window count {true_series.shape[0]}"
            )

        pred_len = true_series.shape[1]
        out_dim = true_series.shape[2]
        seq_len = int(self.config.seq_len)

        true_sum = {}
        pred_sum = {}
        count = {}

        for window_idx, start in enumerate(sample_ids.astype(np.int64)):
            base_time_idx = int(start) + seq_len

            for horizon in range(pred_len):
                time_idx = base_time_idx + horizon
                valid_vec = (
                    valid_mask[window_idx, horizon].astype(np.float64)
                    * np.isfinite(true_series[window_idx, horizon]).astype(np.float64)
                    * np.isfinite(pred_series[window_idx, horizon]).astype(np.float64)
                )

                if not np.any(valid_vec > 0.5):
                    continue

                if time_idx not in count:
                    true_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    pred_sum[time_idx] = np.zeros(out_dim, dtype=np.float64)
                    count[time_idx] = np.zeros(out_dim, dtype=np.float64)

                true_sum[time_idx] += (
                    true_series[window_idx, horizon].astype(np.float64) * valid_vec
                )
                pred_sum[time_idx] += (
                    pred_series[window_idx, horizon].astype(np.float64) * valid_vec
                )
                count[time_idx] += valid_vec

        time_indices = sorted(count)

        if len(time_indices) == 0:
            return (
                np.empty((0, out_dim), dtype=np.float32),
                np.empty((0, out_dim), dtype=np.float32),
                np.asarray([], dtype=np.int64),
            )

        true_rows = []
        pred_rows = []
        kept_time_indices = []

        for time_idx in time_indices:
            valid_dim = count[time_idx] > 0
            if not np.any(valid_dim):
                continue

            true_row = np.full(out_dim, np.nan, dtype=np.float64)
            pred_row = np.full(out_dim, np.nan, dtype=np.float64)

            true_row[valid_dim] = true_sum[time_idx][valid_dim] / count[time_idx][valid_dim]
            pred_row[valid_dim] = pred_sum[time_idx][valid_dim] / count[time_idx][valid_dim]

            true_rows.append(true_row)
            pred_rows.append(pred_row)
            kept_time_indices.append(time_idx)

        true_agg = np.stack(true_rows, axis=0).astype(np.float32)
        pred_agg = np.stack(pred_rows, axis=0).astype(np.float32)

        return true_agg, pred_agg, np.asarray(kept_time_indices, dtype=np.int64)

    def pretrain_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)

        t1 = time.time()

        total_loss = 0.0
        sample_count = 0
        last_aux = None

        for batch_idx, train_batch in enumerate(dataModule.train_data_loader):
            x, _, _, _ = train_batch
            x = x.to(self.config.device, non_blocking=True)

            self.pretrain_optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss, aux = self.model.pretrain_state_prior_loss(x)

                self.pretrain_scaler.scale(loss).backward()
                self.pretrain_scaler.unscale_(self.pretrain_optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.get_state_prior_parameters(),
                    max_norm=1.0,
                )

                self.pretrain_scaler.step(self.pretrain_optimizer)
                self.pretrain_scaler.update()

            else:
                loss, aux = self.model.pretrain_state_prior_loss(x)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.get_state_prior_parameters(),
                    max_norm=1.0,
                )

                self.pretrain_optimizer.step()

            total_loss += loss.item() * x.size(0)
            sample_count += x.size(0)
            last_aux = aux

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataModule.train_data_loader):
                q_mean = aux["q_mean"]

                print(
                    f"[pretrain] Batch {batch_idx + 1}/{len(dataModule.train_data_loader)} - "
                    f"Loss: {loss.item():.6f}, "
                    f"NLL: {aux['pretrain_nll'].item():.6f}, "
                    f"KL: {aux['balance_kl'].item():.4f}, "
                    f"Dom: {aux['dominant_penalty'].item():.4f}, "
                    f"q_mean={['%.3f' % v for v in q_mean.tolist()]}"
                )

        self.eval()
        torch.set_grad_enabled(False)

        t2 = time.time()

        avg_loss = total_loss / max(sample_count, 1)

        print(
            f"[Train-pretrain] Epoch {self.current_epoch} finished | "
            f"Avg NLL: {avg_loss:.6f} | Time Cost: {t2 - t1:.2f}s"
        )

        return avg_loss, t2 - t1, last_aux

    def prepare_retrieval_index(self, train_data, train_loader=None):
        print("*******Constructing the Retrieval Indexes*********")

        # 构建检索库时必须固定顺序
        # 不能直接用 dataModule.train_data_loader，因为它通常 shuffle=True
        index_loader = DataLoader(
            train_data,
            batch_size=int(self.config.bs),
            shuffle=False,
            num_workers=int(getattr(self.config, "num_workers", 0)),
            pin_memory=True,
        )

        # 检索库容量 = 训练集窗口数量
        self.model.construct_index(len(train_data))

        # 如果模型里 retrieval_stride 存在，就同步成数据集真实 stride
        # 这样 retrieval() 里 sample_id // retrieval_stride 才能映射到训练窗口下标
        if hasattr(self.model, "retrieval_stride"):
            self.model.retrieval_stride = int(
                getattr(train_data, "stride", getattr(self.config, "stride", 1))
            )

        write_ptr = 0
        time_now = time.time()
        train_steps = len(index_loader)

        with torch.no_grad():
            for i, batch in enumerate(index_loader):
                batch_x, x_mark, batch_y, sample_ids = batch

                batch_x = batch_x.float().to(self.config.device, non_blocking=True)
                batch_y = batch_y.float().to(self.config.device, non_blocking=True)

                batch_size = batch_x.size(0)

                # 这里才是检索库真正的写入下标：0,1,2,...,N-1
                index_ids = torch.arange(
                    write_ptr,
                    write_ptr + batch_size,
                    device=self.config.device,
                    dtype=torch.long,
                )

                self.model.add_key_value(
                    batch_x,
                    batch_y[:, -self.config.pred_len:, :],
                    index_ids,
                )

                write_ptr += batch_size

                if (i + 1) % 100 == 0 or (i + 1) == train_steps:
                    speed = (time.time() - time_now) / max(i + 1, 1)
                    left_time = speed * max(train_steps - i - 1, 0)

                    print(
                        f"\titers: {i + 1}/{train_steps}, "
                        f"speed: {speed:.4f}s/iter; "
                        f"left time: {left_time:.4f}s"
                    )

        print(
            "*******Finishing the Retrieval Indexes | "
            f"index={self.model.index}, capacity={len(train_data)}, "
            f"retrieval_stride={getattr(self.model, 'retrieval_stride', None)}"
            " *********"
        )

        self.model.value_permute = self.model.values.permute(2, 0, 1)

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
                        output, aux_loss = self.forward(
                            x,
                            x_mark,
                            dec_input,
                            None,
                            sample_ids,
                            mode,
                        )

                        pred_scaled, real_scaled = self._prepare_scaled_prediction_and_label(
                            output["point_pred"],
                            label,
                            dataModule,
                        )

                        label_mask = self._prepare_label_mask(
                            x_mark,
                            label,
                            dataModule,
                        )

                        main_loss = self._compute_masked_supervised_loss(
                            x,
                            pred_scaled.float(),
                            real_scaled.float(),
                            label_mask.float(),
                        )

                        total_loss_value = main_loss + aux_loss

                    scaler.scale(total_loss_value).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    output, aux_loss = self.forward(
                        x,
                        x_mark,
                        dec_input,
                        None,
                        sample_ids,
                        mode,
                    )

                    pred_scaled, real_scaled = self._prepare_scaled_prediction_and_label(
                        output["point_pred"],
                        label,
                        dataModule,
                    )

                    label_mask = self._prepare_label_mask(
                        x_mark,
                        label,
                        dataModule,
                    )

                    main_loss = self._compute_masked_supervised_loss(
                        x,
                        pred_scaled.float(),
                        real_scaled.float(),
                        label_mask.float(),
                    )

                    total_loss_value = main_loss + aux_loss

                    if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataModule.train_data_loader):
                        beta_mean_item = 0.0
                        q_mean_str = ""
                        q_stat_str = ""
                        q_sample_str = ""
                        alpha_str = ""

                        if hasattr(self.model, "latest_aux_dict"):
                            latest_aux = self.model.latest_aux_dict

                            if "beta_mean" in latest_aux:
                                beta_mean_item = float(latest_aux["beta_mean"].item())

                            if "state_probs" in latest_aux:
                                q_all = latest_aux["state_probs"].detach().cpu()

                                q_mean = q_all.mean(dim=0)
                                q_mean_str = ", Q_mean=[" + ", ".join(
                                    f"{v:.3f}" for v in q_mean.tolist()
                                ) + "]"

                                q_max, q_assign = q_all.max(dim=1)
                                qmax_mean = q_max.mean().item()

                                entropy = -(
                                    q_all * q_all.clamp_min(1e-8).log()
                                ).sum(dim=1)

                                entropy_mean = entropy.mean().item()
                                high_conf_ratio = (q_max > 0.7).float().mean().item()

                                q_stat_str = (
                                    f", QmaxMean={qmax_mean:.3f}"
                                    f", QEntropyMean={entropy_mean:.3f}"
                                    f", HighConf(>0.7)={high_conf_ratio:.3f}"
                                )

                                show_n = min(3, q_all.size(0))
                                sample_msgs = []

                                for i in range(show_n):
                                    qi = q_all[i]
                                    qmax_i = q_max[i].item()
                                    assign_i = q_assign[i].item()

                                    sid = sample_ids[i].item() if sample_ids is not None else i

                                    sample_msgs.append(
                                        f"id={sid}, q=[{', '.join(f'{v:.3f}' for v in qi.tolist())}], "
                                        f"argmax={assign_i}, qmax={qmax_i:.3f}"
                                    )

                                q_sample_str = ", Samples: " + " | ".join(sample_msgs)

                            if "state_alpha" in latest_aux:
                                a = latest_aux["state_alpha"].detach().cpu()
                                alpha_str = ", alpha=[" + ", ".join(
                                    f"{v:.2f}" for v in a.tolist()
                                ) + "]"

                        print(
                            f"[{stage}] Batch {batch_idx + 1}/{len(dataModule.train_data_loader)} - "
                            f"Main Loss: {main_loss.item():.6f}, "
                            f"Aux Loss: {aux_loss.item():.6f}, "
                            f"Beta Mean: {beta_mean_item:.6f}"
                            f"{q_mean_str}"
                            f"{q_stat_str}"
                            f"{q_sample_str}"
                            f"{alpha_str}, "
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

    def valid(self, dataModule, stage="backbone"):
        self.eval()
        torch.set_grad_enabled(False)

        dataloader = dataModule.val_data_loader

        preds = []
        reals = []
        ids = []
        masks = []

        mode = "valid" if stage == "backbone" else "gate_valid"

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
                sample_ids = sample_ids.to(self.config.device).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                output, _ = self.forward(
                    x,
                    x_mark,
                    dec_input,
                    None,
                    sample_ids,
                    mode,
                )

                label_mask = self._prepare_label_mask(
                    x_mark,
                    label,
                    dataModule,
                )

                reals.append(label)
                preds.append(output["point_pred"])
                ids.append(sample_ids)
                masks.append(label_mask)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        ids = torch.cat(ids, dim=0)
        masks = torch.cat(masks, dim=0)

        pred_raw, real_raw = self._restore_to_raw(
            preds,
            reals,
            dataModule,
            ids,
        )

        raw_eval_mask = self._recover_level_mask_from_diff_mask(masks)

        real_agg, pred_agg, _ = self.aggregate_series_by_time(
            real_raw,
            pred_raw,
            ids,
            valid_mask=raw_eval_mask,
        )

        return ErrorMetrics(real_agg, pred_agg, self.config, "valid")


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
                sample_ids = sample_ids.to(self.config.device).long()

                dec_input = torch.zeros_like(label[:, -self.pred_len:, :]).float()
                dec_input = torch.cat(
                    [label[:, :self.label_len, :], dec_input],
                    dim=1,
                ).float().to(self.config.device)

                output, _ = self.forward(
                    x,
                    x_mark,
                    dec_input,
                    None,
                    sample_ids,
                    mode="test",
                )

                label_mask = self._prepare_label_mask(
                    x_mark,
                    label,
                    dataModule,
                )

                reals.append(label)
                preds.append(output["point_pred"])
                ids.append(sample_ids)
                masks.append(label_mask)

        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        ids = torch.cat(ids, dim=0)
        masks = torch.cat(masks, dim=0)

        pred_raw, real_raw = self._restore_to_raw(
            preds,
            reals,
            dataModule,
            ids,
        )

        raw_eval_mask = self._recover_level_mask_from_diff_mask(masks)

        dataset_name = getattr(self.config, "dataset", "Dataset")
        model_name = getattr(self.config, "model", "Model")

        data_file = getattr(self.config, "data_file", "data")
        data_tag = os.path.splitext(os.path.basename(str(data_file)))[0]

        target_col = int(getattr(self.config, "target_col", 0))
        original_target_col = int(getattr(self.config, "original_target_col", target_col))

        save_dir = os.path.join(
            "./draw",
            dataset_name,
            data_tag,
            model_name,
            f"PL{self.config.pred_len}_DM{self.config.d_model}",
            f"TC{original_target_col}",
        )

        save_path = os.path.join(save_dir, "test_raw.csv")
        aggregated_save_path = os.path.join(save_dir, "test_agg.csv")

        self.save_single_model_result(
            true_series=real_raw,
            pred_series=pred_raw,
            save_path=save_path,
        )

        self.save_aggregated_model_result(
            true_series=real_raw,
            pred_series=pred_raw,
            sample_ids=ids,
            save_path=aggregated_save_path,
        )

        real_agg, pred_agg, _ = self.aggregate_series_by_time(
            real_raw,
            pred_raw,
            ids,
            valid_mask=raw_eval_mask,
        )

        return ErrorMetrics(real_agg, pred_agg, self.config, "test")

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

        gate_epochs = getattr(config, "gate_epochs", 5)

        if not retrain_required:
            try:
                sum_time = pickle.load(
                    open(f"./results/metrics/" + log.filename + ".pkl", "rb")
                )["train_time"][runId]

                model.load_state_dict(
                    torch.load(model_path, weights_only=True, map_location="cpu")
                )

                model.setup_optimizer(config)

                train_data = datamodule.train_data_loader.dataset
                train_loader = datamodule.train_data_loader

                model.prepare_retrieval_index(train_data, train_loader)

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

            train_data = datamodule.train_data_loader.dataset
            train_loader = datamodule.train_data_loader

            model.prepare_retrieval_index(train_data, train_loader)

            gate_time = []
            gate_trained = False

            if gate_epochs > 0:
                model.model.freeze_backbone_for_gate()
                model.setup_gate_optimizer(config)

                best_gate_metric = float("inf")
                best_gate_state = copy.deepcopy(model.state_dict())

                gate_patience = getattr(config, "gate_patience", config.patience)
                gate_wait = 0

                for gate_epoch in trange(gate_epochs, desc="Gate Training"):
                    model.current_epoch = gate_epoch + 1

                    train_loss_gate, time_cost_gate = model.train_one_epoch(
                        datamodule,
                        stage="gate",
                    )

                    gate_time.append(time_cost_gate)

                    valid_error_gate = model.valid(datamodule, stage="gate")
                    gate_metric = valid_error_gate[config.monitor_metric]

                    model.gate_scheduler.step(gate_metric)

                    print(
                        f"[Gate] Current LR: {model.gate_optimizer.param_groups[0]['lr']:.6g}"
                    )

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
                gate_trained = True

            model.model.mark_gate_ready(gate_trained)
            model.model.unfreeze_all()

            sum_time = sum(train_time[: monitor.best_epoch]) + sum(gate_time)

            results = model.test(datamodule)

            log.show_test_error(runId, monitor, results, sum_time)

            torch.save(model.state_dict(), model_path)

            log(f"Model parameters saved to {model_path}")

        results["train_time"] = sum_time

        return results

    def RunOnce(self, config, runId, model, datamodule, log):
        pretrain_epochs = int(getattr(config, "pretrain_epochs", 20))
        freeze_after_pretrain = bool(
            getattr(config, "freeze_prior_after_pretrain", False)
        )

        enable_pretrain = (
            self._need_retrain(config, runId, log)
            and pretrain_epochs > 0
            and config.model == "net"
        )

        if enable_pretrain:
            print("*******State Prior Pretraining*******")

            self.setup_pretrain_optimizer(config)

            best_loss = float("inf")
            best_state = copy.deepcopy(self.model.state_dict())

            for epoch in trange(pretrain_epochs, desc="State Prior Pretraining"):
                self.current_epoch = epoch + 1

                t_start = float(
                    getattr(
                        config,
                        "state_prior_temperature_start",
                        getattr(config, "state_prior_temperature", 1.0),
                    )
                )

                t_end = float(
                    getattr(config, "state_prior_temperature_end", 0.6)
                )

                if pretrain_epochs > 1:
                    ratio = epoch / float(pretrain_epochs - 1)
                else:
                    ratio = 1.0

                self.model.state_prior.temperature = t_start + (t_end - t_start) * ratio

                train_nll, _, last_aux = self.pretrain_one_epoch(datamodule)

                self.pretrain_scheduler.step(train_nll)

                print(
                    f"[Pretrain] Current LR: {self.pretrain_optimizer.param_groups[0]['lr']:.6g}, "
                    f"Temp: {self.model.state_prior.temperature:.4f}"
                )

                if train_nll < best_loss:
                    best_loss = train_nll
                    best_state = copy.deepcopy(self.model.state_dict())

            self.model.load_state_dict(best_state)

            qmax = 0.0

            collapse_threshold = min(
                0.95,
                float(getattr(config, "state_dom_cap", 0.8)) + 0.1,
            )

            if "last_aux" in locals() and last_aux is not None and "q_mean" in last_aux:
                qmax = float(last_aux["q_mean"].max().item())

            if freeze_after_pretrain and qmax < collapse_threshold:
                self.model.freeze_state_prior()
                print("[Pretrain] state prior is frozen for backbone training")
            else:
                self.model.unfreeze_state_prior()
                print(
                    f"[Pretrain] keep state prior trainable in backbone stage "
                    f"(qmax={qmax:.3f})"
                )

        return self._run_backbone_gate_once(
            config,
            runId,
            model,
            datamodule,
            log,
        )
