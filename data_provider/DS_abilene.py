# Author  : mkw (adapted)
# Desc    : Abilene dataset loader (standard time-series), 7:1:2 split

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


class AbileneDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, config):
        self.x = x
        self.y = y
        self.id_offset = int(getattr(config, "id_offset", 0))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        # x_mark is unused by ExtremeLSTMMemo; keep a placeholder for API compatibility.
        x_mark = np.zeros((x.shape[0], 1), dtype=np.float32)
        sample_id = np.int64(self.id_offset + idx)
        return x.astype(np.float32), x_mark, y.astype(np.float32), sample_id


@dataclass
class SplitSizes:
    train: int
    val: int
    test: int


class DS:
    """Abilene dataset processing with ratio split 7:1:2."""

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

        self.mean = None
        self.std = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self._load_and_build()

    def _load_csv(self) -> np.ndarray:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Abilene csv not found: {self.data_path}")
        data = np.loadtxt(self.data_path, delimiter=",")
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")
        return data.astype(np.float32)

    def _split_ratio(self, n: int) -> Tuple[SplitSizes, Tuple[int, int]]:
        train_size = int(n * 0.7)
        val_size = int(n * 0.1)
        test_size = n - train_size - val_size
        return SplitSizes(train_size, val_size, test_size), (train_size, train_size + val_size)

    @staticmethod
    def _normalize(train: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return (data - mean) / std, mean, std

    def _make_windows(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total = arr.shape[0] - self.seq_len - self.pred_len + 1
        if total <= 0:
            raise ValueError("Sequence too short for given seq_len/pred_len")

        x = np.lib.stride_tricks.sliding_window_view(arr, (self.seq_len, arr.shape[1]))[:, 0, :, :]
        y = np.lib.stride_tricks.sliding_window_view(arr[self.seq_len :], (self.pred_len, arr.shape[1]))[:, 0, :, :]

        if self.stride > 1:
            x = x[:: self.stride]
            y = y[:: self.stride]

        total = min(x.shape[0], y.shape[0])
        return x[:total], y[:total]

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
        self.train_data_loader = DataLoader(
            AbileneDataset(x_train, y_train, self.config),
            batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        self.val_data_loader = DataLoader(
            AbileneDataset(x_val, y_val, self.config),
            batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        self.test_data_loader = DataLoader(
            AbileneDataset(x_test, y_test, self.config),
            batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        print(
            f"[Abilene] split: train={split_sizes.train}, val={split_sizes.val}, test={split_sizes.test}"
        )
        print(
            "[Abilene] windows:",
            x_train.shape,
            x_val.shape,
            x_test.shape,
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
