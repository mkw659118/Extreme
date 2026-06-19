# Author  : mkw
# Desc    : Abilene dataset loader with raw value standardization and artificial missing masks
# Shape   : raw data [T, C] = [3000, 144]

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_mask: np.ndarray,
        config,
        id_offset: int = 0,
        stride: int = 96,
    ):
        del config
        self.x = x
        self.y = y
        self.y_mask = y_mask
        self.id_offset = int(id_offset)
        self.stride = int(stride)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        y_mask = self.y_mask[idx]

        # Same convention as DARNet: x_mark carries future-label validity.
        x_mark = y_mask.astype(np.float32)
        sample_id = np.int64(self.id_offset + idx * self.stride)

        return (
            x.astype(np.float32),
            x_mark,
            y.astype(np.float32),
            sample_id,
        )


@dataclass
class SplitSizes:
    train: int
    val: int
    test: int


class DS:
    """Raw-value z-score dataset with missing-aware input corruption.

    Baselines do not use first-order differences. Artificial missingness is
    therefore applied directly to raw-value inputs before z-score
    standardization. Future labels and their masks are built from the original
    observation mask, so input missing-rate experiments do not change the
    supervised target set.
    """

    def __init__(self, config, trainX=None):
        del trainX

        self.config = config
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.stride = int(getattr(self.config, "stride", 1))

        self.mask_zero_as_missing = bool(
            getattr(self.config, "mask_zero_as_missing", True)
        )
        setattr(self.config, "mask_zero_as_missing", self.mask_zero_as_missing)

        self.data_root = getattr(self.config, "path", "./datasets")
        self.dataset = getattr(self.config, "dataset", "Abilene")
        self.data_file = getattr(self.config, "data_file", "Abilene_single.csv")
        self.data_path = os.path.join(
            self.data_root,
            self.dataset,
            self.data_file,
        )

        self.target_col = 0
        self.target_dim = int(getattr(self.config, "target_dim", 0) or 0)
        self.num_vars = None

        setattr(self.config, "target_col", self.target_col)
        setattr(self.config, "target_dim", self.target_dim)

        self.raw_data = None
        self.raw_mask = None
        self.mean = None
        self.std = None

        self.tail_q90 = 0.0
        self.raw_q90 = 0.0
        self.raw_q99 = 0.0

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self._load_and_build()

    def _load_csv(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            fallback_path = os.path.join(self.data_root, self.data_file)
            if os.path.exists(fallback_path):
                self.data_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"CSV file not found: {self.data_path} or {fallback_path}"
                )

        data = np.loadtxt(self.data_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [T, C], got shape {data.shape}")
        if np.isinf(data).any():
            raise ValueError("Inf found in csv.")

        self.num_vars = int(data.shape[1])
        self.target_col = 0
        self.target_dim = self.num_vars
        setattr(self.config, "enc_in", self.num_vars)
        setattr(self.config, "dec_in", self.num_vars)
        setattr(self.config, "c_out", self.num_vars)
        setattr(self.config, "out_dim", self.num_vars)
        setattr(self.config, "num_vars", self.num_vars)
        setattr(self.config, "target_col", self.target_col)
        setattr(self.config, "target_dim", self.target_dim)
        setattr(self.config, "target_cols", "all")

        if self.target_dim != self.num_vars:
            raise ValueError(
                f"Baseline multi predicts all variables, but target_dim={self.target_dim} "
                f"and num_vars={self.num_vars}."
            )

        return data.astype(np.float32)

    def _split_ratio(self, n: int) -> Tuple[SplitSizes, Tuple[int, int]]:
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
        test_size = n - train_size - val_size
        train_end = train_size
        val_end = train_size + val_size
        return SplitSizes(train_size, val_size, test_size), (train_end, val_end)

    def _build_observation_mask(self, data: np.ndarray) -> np.ndarray:
        mask = np.isfinite(data)
        if self.mask_zero_as_missing:
            mask = mask & (data != 0)
        return mask.astype(np.float32)

    def _get_artificial_missing_splits(self) -> set[str]:
        splits = getattr(
            self.config,
            "artificial_missing_splits",
            "train,val,test",
        )
        if splits is None:
            return set()
        if isinstance(splits, (list, tuple, set)):
            values = splits
        else:
            values = str(splits).replace(";", ",").replace("|", ",").split(",")
        return {str(item).strip().lower() for item in values if str(item).strip()}

    def _apply_random_artificial_missing(
        self,
        mask: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> np.ndarray:
        rate = float(getattr(self.config, "artificial_missing_rate", 0.0))
        if rate <= 0.0:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)
        if rate >= 1.0:
            raise ValueError(
                "artificial_missing_rate must be in [0, 1). "
                f"Got {rate}."
            )

        splits = self._get_artificial_missing_splits()
        if not splits:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)

        seed = int(getattr(self.config, "artificial_missing_seed", 2026))
        target_only = bool(
            getattr(self.config, "artificial_missing_target_only", False)
        )
        rng = np.random.default_rng(seed)

        split_ranges = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "valid": (train_end, val_end),
            "validation": (train_end, val_end),
            "test": (val_end, mask.shape[0]),
            "all": (0, mask.shape[0]),
        }

        candidate = np.zeros_like(mask, dtype=bool)
        for split in splits:
            if split not in split_ranges:
                raise ValueError(
                    "Unknown artificial_missing_splits item "
                    f"'{split}'. Use train,val,test or all."
                )
            start, end = split_ranges[split]
            candidate[start:end, :] = True

        if target_only:
            # In the multi-target baseline every original variable is a target,
            # so target_only keeps all observed variable positions.
            candidate = candidate.copy()

        candidate &= mask > 0.5
        candidate_idx = np.flatnonzero(candidate.reshape(-1))
        total_candidates = int(candidate_idx.size)
        if total_candidates == 0:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)

        remove_count = int(round(total_candidates * rate))
        remove_count = min(max(remove_count, 0), total_candidates)
        if remove_count == 0:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)

        removed_flat = rng.choice(candidate_idx, size=remove_count, replace=False)
        new_mask = mask.reshape(-1).copy()
        new_mask[removed_flat] = 0.0
        new_mask = new_mask.reshape(mask.shape).astype(np.float32)

        actual_rate = remove_count / total_candidates
        setattr(self.config, "artificial_missing_actual_rate", actual_rate)
        print(
            "[Artificial Missing] "
            f"rate={rate:.4f}, actual_rate={actual_rate:.4f}, "
            f"removed={remove_count}/{total_candidates}, "
            f"splits={','.join(sorted(splits))}, "
            f"target_only={target_only}, seed={seed}"
        )

        return new_mask

    @classmethod
    def _zscore_normalize(
        cls,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mask is None:
            mask = np.isfinite(data).astype(np.float32)
        else:
            mask = (mask > 0.5).astype(np.float32)

        safe_data = np.where(mask > 0.5, data, 0.0).astype(np.float32)

        if mean is None or std is None:
            count = mask.sum(axis=0).astype(np.float32)
            safe_count = np.maximum(count, 1.0)
            mean_est = safe_data.sum(axis=0) / safe_count
            centered = np.where(mask > 0.5, safe_data - mean_est.reshape(1, -1), 0.0)
            var_est = (centered ** 2).sum(axis=0) / safe_count
            std_est = np.sqrt(var_est)

            mean_est = np.where(count > 0, mean_est, 0.0)
            std_est = np.where((count > 0) & (std_est > 0), std_est, 1.0)

            if mean is None:
                mean = mean_est
            if std is None:
                std = std_est

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where((std == 0) | np.isnan(std), 1.0, std)

        norm_valid = (safe_data - mean.reshape(1, -1)) / std.reshape(1, -1)
        norm = np.where(mask > 0.5, norm_valid, 0.0)

        return norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def _fit_normalizer(self, train_raw: np.ndarray, train_mask: np.ndarray):
        _, self.mean, self.std = self._zscore_normalize(train_raw, train_mask)

    def _transform_raw_values(self, raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        norm, _, _ = self._zscore_normalize(
            raw,
            mask=mask,
            mean=self.mean,
            std=self.std,
        )
        return norm

    def _make_windows(
        self,
        input_norm_data: np.ndarray,
        label_norm_data: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
    ):
        if label_norm_data is None:
            label_norm_data = input_norm_data
        if label_mask is None:
            label_mask = np.ones_like(label_norm_data, dtype=np.float32)

        if (
            input_norm_data.shape != label_norm_data.shape
            or label_norm_data.shape != label_mask.shape
        ):
            raise ValueError(
                "input_norm_data, label_norm_data and label_mask must have "
                f"same shape, got {input_norm_data.shape}, "
                f"{label_norm_data.shape}, {label_mask.shape}"
            )

        total = input_norm_data.shape[0] - self.seq_len - self.pred_len + 1
        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={input_norm_data.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []
        y_mask_list = []

        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len
            x = input_norm_data[start : start + self.seq_len, :]
            y = label_norm_data[y_start:y_end, :]
            y_valid = label_mask[y_start:y_end, :]

            x_list.append(x)
            y_list.append(y)
            y_mask_list.append(y_valid)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        y_mask = np.stack(y_mask_list, axis=0).astype(np.float32)

        return x, y, y_mask

    def _load_and_build(self):
        raw_loaded = self._load_csv()
        _, (train_end, val_end) = self._split_ratio(len(raw_loaded))

        full_mask = self._build_observation_mask(raw_loaded)
        input_mask = self._apply_random_artificial_missing(
            full_mask,
            train_end,
            val_end,
        )

        data = np.where(full_mask > 0.5, raw_loaded, 0.0).astype(np.float32)
        input_data = np.where(input_mask > 0.5, raw_loaded, 0.0).astype(np.float32)
        self.raw_data = data
        self.raw_mask = full_mask.astype(np.float32)

        train_raw = data[:train_end]
        train_mask = full_mask[:train_end]
        train_input_raw = input_data[:train_end]
        train_input_mask = input_mask[:train_end]

        val_start = max(0, train_end - self.seq_len)
        test_start = max(0, val_end - self.seq_len)

        val_raw = data[val_start:val_end]
        val_mask = full_mask[val_start:val_end]
        val_input_raw = input_data[val_start:val_end]
        val_input_mask = input_mask[val_start:val_end]

        test_raw = data[test_start:]
        test_mask = full_mask[test_start:]
        test_input_raw = input_data[test_start:]
        test_input_mask = input_mask[test_start:]

        self._fit_normalizer(train_raw, train_mask)

        train_norm = self._transform_raw_values(train_raw, train_mask)
        val_norm = self._transform_raw_values(val_raw, val_mask)
        test_norm = self._transform_raw_values(test_raw, test_mask)

        train_input_norm = self._transform_raw_values(train_input_raw, train_input_mask)
        val_input_norm = self._transform_raw_values(val_input_raw, val_input_mask)
        test_input_norm = self._transform_raw_values(test_input_raw, test_input_mask)

        x_train, y_train, y_train_mask = self._make_windows(
            train_input_norm,
            label_norm_data=train_norm,
            label_mask=train_mask,
        )
        x_val, y_val, y_val_mask = self._make_windows(
            val_input_norm,
            label_norm_data=val_norm,
            label_mask=val_mask,
        )
        x_test, y_test, y_test_mask = self._make_windows(
            test_input_norm,
            label_norm_data=test_norm,
            label_mask=test_mask,
        )

        train_value_raw = self.inverse_value_norm(y_train)
        all_value_vals = np.abs(train_value_raw[y_train_mask > 0.5]).reshape(-1)
        all_value_vals = all_value_vals[np.isfinite(all_value_vals)]
        self.tail_q90 = float(np.quantile(all_value_vals, 0.90)) if len(all_value_vals) > 0 else 0.0
        setattr(self.config, "tail_q90", self.tail_q90)
        print(f"[Tail Threshold] |value| q90={self.tail_q90:.6f}")

        raw_y_list = []
        raw_y_mask_list = []
        total = train_raw.shape[0] - self.seq_len - self.pred_len + 1
        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len
            raw_y = train_raw[y_start:y_end, :]
            raw_y_mask = train_mask[y_start:y_end, :]
            raw_y_list.append(raw_y)
            raw_y_mask_list.append(raw_y_mask)

        if len(raw_y_list) > 0:
            all_raw_vals = np.stack(raw_y_list, axis=0).astype(np.float32)
            all_raw_masks = np.stack(raw_y_mask_list, axis=0).astype(np.float32)
            all_raw_vals = all_raw_vals[all_raw_masks > 0.5].reshape(-1)
            all_raw_vals = all_raw_vals[np.isfinite(all_raw_vals)]
            if len(all_raw_vals) > 0:
                self.raw_q90 = float(np.quantile(all_raw_vals, 0.90))
                self.raw_q99 = float(np.quantile(all_raw_vals, 0.99))
            else:
                self.raw_q90 = 0.0
                self.raw_q99 = 0.0

        setattr(self.config, "raw_q90", self.raw_q90)
        setattr(self.config, "raw_q99", self.raw_q99)
        print(
            f"[Raw Threshold] value q90={self.raw_q90:.6f}, "
            f"q99={self.raw_q99:.6f}"
        )

        batch_size = int(self.config.bs)
        num_workers = int(getattr(self.config, "num_workers", 0))

        self.train_data_loader = DataLoader(
            TimeSeriesDataset(
                x_train,
                y_train,
                y_train_mask,
                self.config,
                id_offset=0,
                stride=self.stride,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_data_loader = DataLoader(
            TimeSeriesDataset(
                x_val,
                y_val,
                y_val_mask,
                self.config,
                id_offset=val_start,
                stride=self.stride,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_data_loader = DataLoader(
            TimeSeriesDataset(
                x_test,
                y_test,
                y_test_mask,
                self.config,
                id_offset=test_start,
                stride=self.stride,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"[Dataset] raw shape: {data.shape}, T={data.shape[0]}, C={data.shape[1]}")
        print(f"[Dataset] split points: train_end={train_end}, val_end={val_end}")
        print(
            f"[Dataset] raw split lengths with context: "
            f"train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}"
        )
        print(
            "[Dataset] windows:",
            f"x_train={x_train.shape}, y_train={y_train.shape}, mask_train={y_train_mask.shape}",
            f"x_val={x_val.shape}, y_val={y_val.shape}, mask_val={y_val_mask.shape}",
            f"x_test={x_test.shape}, y_test={y_test.shape}, mask_test={y_test_mask.shape}",
        )
        print(
            f"[Dataset] missing convention: mask_zero_as_missing={self.mask_zero_as_missing}"
        )
        print(
            f"[Dataset] many-to-many target: "
            f"target_cols=all, target_dim={self.target_dim}"
        )
        print(
            f"[Dataset] value norm mean/std shape: "
            f"mean={self.mean.shape}, std={self.std.shape}"
        )

    def inverse_value_norm(self, value_norm: np.ndarray) -> np.ndarray:
        value_norm = np.asarray(value_norm)

        if value_norm.shape[-1] == self.num_vars:
            mean = self.mean
            std = self.std
        elif value_norm.shape[-1] == 1:
            mean = self.mean[self.target_col : self.target_col + 1]
            std = self.std[self.target_col : self.target_col + 1]
        else:
            raise ValueError(
                f"Cannot inverse-normalize values with last dim {value_norm.shape[-1]}; "
                f"expected 1 or num_vars={self.num_vars}."
            )

        return value_norm * std + mean

    def get_train_data_loader(self):
        return self.train_data_loader

    def get_val_data_loader(self):
        return self.val_data_loader

    def get_test_data_loader(self):
        return self.test_data_loader

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_raw_data(self):
        return self.raw_data

    def get_raw_mask(self):
        return self.raw_mask
