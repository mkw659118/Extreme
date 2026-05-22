# Author  : mkw
# Desc    : Abilene dataset loader for multivariate time series, shape [T, C] = [3000, 144]

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


class AbileneDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, config, id_offset: int = 0):
        self.x = x
        self.y = y
        self.id_offset = int(id_offset)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        # ExtremeLSTMMemo 当前没有真正使用 x_mark，这里保留占位
        x_mark = np.zeros((x.shape[0], 1), dtype=np.float32)

        sample_id = np.int64(self.id_offset + idx)

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
    """Abilene multivariate time-series dataset, ratio split 7:1:2."""

    def __init__(self, config, trainX=None):
        del trainX

        self.config = config
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.stride = int(getattr(self.config, "stride", 1))

        self.data_path = getattr(
            self.config,
            "data_path",
            "./datasets/Abilene_12_12_3000_T3000_flat144.csv",
        )

        self.expected_num_vars = int(getattr(self.config, "enc_in", 144))

        self.mean = None
        self.std = None

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self._load_and_build()

    def _load_csv(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Abilene csv not found: {self.data_path}")

        # CSV 没有表头，形状应为 [T, C]，例如 [3000, 144]
        data = np.loadtxt(self.data_path, delimiter=",")

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        if data.shape[1] != self.expected_num_vars:
            raise ValueError(
                f"Variable dimension mismatch: data has {data.shape[1]} variables, "
                f"but config.enc_in={self.expected_num_vars}."
            )

        if np.isnan(data).any():
            raise ValueError("NaN found in Abilene csv.")

        return data.astype(np.float32)

    def _split_ratio(self, n: int) -> Tuple[SplitSizes, Tuple[int, int]]:
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
        test_size = n - train_size - val_size

        train_end = train_size
        val_end = train_size + val_size

        return SplitSizes(train_size, val_size, test_size), (train_end, val_end)

    @staticmethod
    def _normalize(train: np.ndarray, data: np.ndarray):
        # 按变量维度做标准化，mean/std 形状为 [C]
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std = np.where(std == 0, 1.0, std)

        data_norm = (data - mean) / std

        return data_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    def _make_windows(self, arr: np.ndarray):
        """
        arr: [T, C]

        return:
            x: [N, seq_len, C]
            y: [N, pred_len, C]
        """
        total = arr.shape[0] - self.seq_len - self.pred_len + 1

        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={arr.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []

        for start in range(0, total, self.stride):
            x = arr[start : start + self.seq_len]
            y = arr[start + self.seq_len : start + self.seq_len + self.pred_len]

            x_list.append(x)
            y_list.append(y)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)

        return x, y

    def _load_and_build(self):
        data = self._load_csv()

        split_sizes, (train_end, val_end) = self._split_ratio(len(data))

        train_raw = data[:train_end]
        val_raw = data[train_end:val_end]
        test_raw = data[val_end:]

        train_norm, self.mean, self.std = self._normalize(train_raw, train_raw)
        val_norm, _, _ = self._normalize(train_raw, val_raw)
        test_norm, _, _ = self._normalize(train_raw, test_raw)

        x_train, y_train = self._make_windows(train_norm)
        x_val, y_val = self._make_windows(val_norm)
        x_test, y_test = self._make_windows(test_norm)

        batch_size = int(self.config.bs)
        num_workers = int(getattr(self.config, "num_workers", 0))

        self.train_data_loader = DataLoader(
            AbileneDataset(x_train, y_train, self.config, id_offset=0),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.val_data_loader = DataLoader(
            AbileneDataset(x_val, y_val, self.config, id_offset=len(x_train)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.test_data_loader = DataLoader(
            AbileneDataset(x_test, y_test, self.config, id_offset=len(x_train) + len(x_val)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(
            f"[Abilene] raw shape: {data.shape}, "
            f"T={data.shape[0]}, C={data.shape[1]}"
        )
        print(
            f"[Abilene] split: "
            f"train={split_sizes.train}, val={split_sizes.val}, test={split_sizes.test}"
        )
        print(
            "[Abilene] windows:",
            f"x_train={x_train.shape}, y_train={y_train.shape}",
            f"x_val={x_val.shape}, y_val={y_val.shape}",
            f"x_test={x_test.shape}, y_test={y_test.shape}",
        )

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