# Author  : mkw
# Desc    : Abilene dataset loader with raw value standardization only
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
        config,
        id_offset: int = 0,
        stride: int = 96,
    ):
        self.x = x
        self.y = y
        self.id_offset = int(id_offset)
        self.stride = int(stride)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        # 当前模型没有真正使用 x_mark，保留占位
        x_mark = np.zeros((x.shape[0], 4), dtype=np.float32)

        # sample_id 表示这个窗口在原始全序列中的起始位置
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
    """Abilene multivariate time-series dataset with raw-value z-score standardization only."""

    def __init__(self, config, trainX=None):
        del trainX

        self.config = config
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.stride = int(getattr(self.config, "stride", 1))

        self.data_root = getattr(self.config, "path", "./datasets")
        self.dataset = getattr(self.config, "dataset", "Abilene")
        self.data_file = getattr(self.config, "data_file", "Abilene_single.csv")
        
        self.data_path = os.path.join(
            self.data_root,
            self.dataset,
            self.data_file,
        )


        self.target_col = int(getattr(self.config, "target_col", 0))
        self.target_dim = 1
        self.num_vars = None

        setattr(self.config, "target_col", self.target_col)
        setattr(self.config, "target_dim", self.target_dim)

        # 原始数据，评估阶段反标准化或对齐原始时间点时可使用
        self.raw_data = None

        # 标准化参数，形状都是 [C]，只使用训练集拟合
        self.mean = None
        self.std = None

        # 训练集阈值，用于 Tail / level 指标
        self.tail_q90 = 0.0
        self.raw_q90 = 0.0
        self.raw_q99 = 0.0

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self._load_and_build()

    def _load_csv(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"CSV file not found: {self.data_path}")

        data = np.loadtxt(self.data_path, delimiter=",")

        # 单变量数据可能被 np.loadtxt 读成 [T]，统一转成 [T, 1]
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array [T, C], got shape {data.shape}")

        if np.isnan(data).any():
            raise ValueError("NaN found in csv.")

        # 直接从数据本身推断变量数
        self.num_vars = int(data.shape[1])

        # 回写 config，给后面的模型初始化用
        setattr(self.config, "enc_in", self.num_vars)
        setattr(self.config, "num_vars", self.num_vars)

        # 现在才检查 target_col 是否越界
        if not 0 <= self.target_col < self.num_vars:
            raise ValueError(
                f"target_col={self.target_col} is out of range. "
                f"Data has {self.num_vars} variables."
            )

        return data.astype(np.float32)

    def _split_ratio(self, n: int) -> Tuple[SplitSizes, Tuple[int, int]]:
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
        test_size = n - train_size - val_size

        train_end = train_size
        val_end = train_size + val_size

        return SplitSizes(train_size, val_size, test_size), (train_end, val_end)

    @classmethod
    def _zscore_normalize(
        cls,
        data: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对原始值做 Z-score 标准化。

        data: [T, C]
        return:
            norm: [T, C]
            mean: [C]
            std:  [C]
        """
        if mean is None:
            mean = np.nanmean(data, axis=0)
        if std is None:
            std = np.nanstd(data, axis=0)

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        mean = np.where(np.isnan(mean), 0.0, mean)
        std = np.where((std == 0) | np.isnan(std), 1.0, std)

        norm = (data - mean) / std

        return norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def _fit_normalizer(self, train_raw: np.ndarray):
        """只用训练集拟合 mean/std，避免验证集和测试集信息泄漏。"""
        _, self.mean, self.std = self._zscore_normalize(train_raw)

    def _transform_raw_values(self, raw: np.ndarray) -> np.ndarray:
        """使用训练集 mean/std，把原始序列转成标准化序列。"""
        norm, _, _ = self._zscore_normalize(
            raw,
            mean=self.mean,
            std=self.std,
        )
        return norm

    def _make_windows(self, norm_data: np.ndarray):
        """
        norm_data: [T, C]

        return:
            x: [N, seq_len, C]
            y: [N, pred_len, 1]

        x 和 y 都是标准化后的原始值。
        模型预测标准化空间的 y，评估阶段可通过 inverse_value_norm 反标准化回原始值。
        """
        total = norm_data.shape[0] - self.seq_len - self.pred_len + 1

        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={norm_data.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []

        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len
            x = norm_data[start : start + self.seq_len, :]
            y = norm_data[y_start:y_end, self.target_col : self.target_col + 1]

            x_list.append(x)
            y_list.append(y)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)

        return x, y

    def _load_and_build(self):
        data = self._load_csv()
        self.raw_data = data

        _, (train_end, val_end) = self._split_ratio(len(data))

        train_raw = data[:train_end]

        # 验证集和测试集保留前 seq_len 个历史点作为输入上下文
        val_start = max(0, train_end - self.seq_len)
        test_start = max(0, val_end - self.seq_len)

        val_raw = data[val_start:val_end]
        test_raw = data[test_start:]

        # 只用训练集拟合标准化参数
        self._fit_normalizer(train_raw)

        train_norm = self._transform_raw_values(train_raw)
        val_norm = self._transform_raw_values(val_raw)
        test_norm = self._transform_raw_values(test_raw)

        x_train, y_train = self._make_windows(train_norm)
        x_val, y_val = self._make_windows(val_norm)
        x_test, y_test = self._make_windows(test_norm)

        # ===== 计算训练集目标值 |value| q90（用于 Tail 指标） =====
        # y_train 是标准化后的目标值，因此先反标准化成原始值，
        # 再把所有窗口、所有预测步展开后计算分位数。
        train_value_raw = self.inverse_value_norm(y_train)
        all_value_vals = np.abs(train_value_raw).reshape(-1)
        all_value_vals = all_value_vals[np.isfinite(all_value_vals)]

        if len(all_value_vals) > 0:
            self.tail_q90 = float(np.quantile(all_value_vals, 0.90))
        else:
            self.tail_q90 = 0.0

        setattr(self.config, "tail_q90", self.tail_q90)
        print(f"[Tail Threshold] |value| q90={self.tail_q90:.6f}")

        # ===== 计算训练集点级原始值 q90/q99（用于 level 指标） =====
        # y_train 对应的原始值区间是：
        # raw[start + seq_len : start + seq_len + pred_len]
        raw_y_list = []
        total = train_raw.shape[0] - self.seq_len - self.pred_len + 1

        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len
            raw_y = train_raw[y_start:y_end, self.target_col : self.target_col + 1]
            raw_y_list.append(raw_y)

        if len(raw_y_list) > 0:
            all_raw_vals = np.stack(raw_y_list, axis=0).astype(np.float32).reshape(-1)
            all_raw_vals = all_raw_vals[np.isfinite(all_raw_vals)]

            if len(all_raw_vals) > 0:
                self.raw_q90 = float(np.quantile(all_raw_vals, 0.90))
                self.raw_q99 = float(np.quantile(all_raw_vals, 0.99))
            else:
                self.raw_q90 = 0.0
                self.raw_q99 = 0.0
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
                self.config,
                id_offset=test_start,
                stride=self.stride,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(
            f"[Dataset] raw shape: {data.shape}, "
            f"T={data.shape[0]}, C={data.shape[1]}"
        )
        print(
            f"[Dataset] split points: "
            f"train_end={train_end}, val_end={val_end}"
        )
        print(
            f"[Dataset] raw split lengths with context: "
            f"train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}"
        )
        print(
            "[Dataset] windows:",
            f"x_train={x_train.shape}, y_train={y_train.shape}",
            f"x_val={x_val.shape}, y_val={y_val.shape}",
            f"x_test={x_test.shape}, y_test={y_test.shape}",
        )
        print(
            f"[Dataset] many-to-one target: "
            f"target_col={self.target_col}, target_dim={self.target_dim}"
        )
        print(
            f"[Dataset] value norm mean/std shape: "
            f"mean={self.mean.shape}, std={self.std.shape}"
        )

    def inverse_value_norm(self, value_norm: np.ndarray) -> np.ndarray:
        """
        将标准化后的值反标准化回原始值。

        value_norm: [..., C] or [..., 1]
        return:    [..., C] or [..., 1]
        """
        value_norm = np.asarray(value_norm)

        if value_norm.shape[-1] == self.target_dim:
            mean = self.mean[self.target_col : self.target_col + 1]
            std = self.std[self.target_col : self.target_col + 1]
        else:
            mean = self.mean
            std = self.std

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
