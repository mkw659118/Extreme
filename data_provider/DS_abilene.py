# Author  : mkw
# Desc    : Abilene dataset loader with first-order difference normalization
# Shape   : raw data [T, C] = [3000, 144]

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


class AbileneDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config,
        id_offset: int = 0,
        stride: int = 1,
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
        x_mark = np.zeros((x.shape[0], 1), dtype=np.float32)

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
    """Abilene multivariate time-series dataset with first-order diff normalization."""

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

        # 原始数据，评估阶段累加还原会用到
        self.raw_data = None

        # 差分归一化参数，形状都是 [C]
        self.mean = None
        self.std = None
        self.mini = 0.0

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
    def _first_order_diff(data: np.ndarray) -> np.ndarray:
        """
        模拟你原来的 r_log_std_normalization 做法：

        原单变量逻辑：
            data1 = data[1:]
            data2[i] = data[i+1] - data[i]
            c = np.array([1] + data2)

        多变量版本：
            data: [T, C]
            diff: [T, C]
            diff[0, :] = 1
            diff[t, :] = data[t, :] - data[t-1, :]
        """
        diff = np.zeros_like(data, dtype=np.float32)
        diff[0, :] = 1.0
        diff[1:, :] = data[1:, :] - data[:-1, :]
        return diff

    def _fit_diff_normalizer(self, train_raw: np.ndarray):
        """
        只用训练集拟合差分归一化参数，避免验证/测试泄漏。
        """
        train_diff = self._first_order_diff(train_raw)

        mean = np.nanmean(train_diff, axis=0)
        std = np.nanstd(train_diff, axis=0)

        std = np.where((std == 0) | np.isnan(std), 1.0, std)
        mean = np.where(np.isnan(mean), 0.0, mean)

        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.mini = 0.0

    def _transform_diff(self, raw: np.ndarray) -> np.ndarray:
        """
        使用训练集 mean/std，把原始序列转成标准化一阶差分序列。
        """
        diff = self._first_order_diff(raw)
        norm = (diff - self.mean) / self.std
        return norm.astype(np.float32)

    def _make_windows(self, diff_norm: np.ndarray):
        """
        diff_norm: [T, C]

        return:
            x: [N, seq_len, C]
            y: [N, pred_len, C]

        x 和 y 都是一阶差分归一化后的值。
        模型预测 y，评估阶段再反归一化并累加还原。
        """
        total = diff_norm.shape[0] - self.seq_len - self.pred_len + 1

        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={diff_norm.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []

        for start in range(0, total, self.stride):
            x = diff_norm[start : start + self.seq_len]
            y = diff_norm[
                start + self.seq_len :
                start + self.seq_len + self.pred_len
            ]

            x_list.append(x)
            y_list.append(y)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)

        return x, y

    def _load_and_build(self):
        data = self._load_csv()
        self.raw_data = data

        split_sizes, (train_end, val_end) = self._split_ratio(len(data))

        train_raw = data[:train_end]

        # 验证集和测试集保留前 seq_len 个历史点作为输入上下文
        val_start = max(0, train_end - self.seq_len)
        test_start = max(0, val_end - self.seq_len)

        val_raw = data[val_start:val_end]
        test_raw = data[test_start:]

        # 只用训练集拟合差分归一化参数
        self._fit_diff_normalizer(train_raw)

        train_norm = self._transform_diff(train_raw)
        val_norm = self._transform_diff(val_raw)
        test_norm = self._transform_diff(test_raw)

        x_train, y_train = self._make_windows(train_norm)
        x_val, y_val = self._make_windows(val_norm)
        x_test, y_test = self._make_windows(test_norm)

        batch_size = int(self.config.bs)
        num_workers = int(getattr(self.config, "num_workers", 0))

        self.train_data_loader = DataLoader(
            AbileneDataset(
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
            AbileneDataset(
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
            AbileneDataset(
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
            f"[Abilene] raw shape: {data.shape}, "
            f"T={data.shape[0]}, C={data.shape[1]}"
        )
        print(
            f"[Abilene] split points: "
            f"train_end={train_end}, val_end={val_end}"
        )
        print(
            f"[Abilene] raw split lengths with context: "
            f"train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}"
        )
        print(
            "[Abilene] windows:",
            f"x_train={x_train.shape}, y_train={y_train.shape}",
            f"x_val={x_val.shape}, y_val={y_val.shape}",
            f"x_test={x_test.shape}, y_test={y_test.shape}",
        )
        print(
            f"[Abilene] diff norm mean/std shape: "
            f"mean={self.mean.shape}, std={self.std.shape}"
        )

    def inverse_diff_norm(self, diff_norm: np.ndarray) -> np.ndarray:
        """
        把标准化差分还原成原始差分。

        diff_norm: [..., C]
        return:    [..., C]
        """
        return diff_norm * self.std + self.mean

    def recover_level_from_diff(
        self,
        diff_norm: np.ndarray,
        sample_ids: np.ndarray,
    ) -> np.ndarray:
        """
        将预测的标准化差分累加还原成原始值。

        diff_norm:  [B, pred_len, C]
        sample_ids: [B]，每个窗口在原始序列中的起始位置

        对每个样本：
            anchor = raw_data[start + seq_len - 1]
            pred_raw = anchor + cumsum(pred_diff_raw)
        """
        diff_raw = self.inverse_diff_norm(diff_norm)

        sample_ids = np.asarray(sample_ids).astype(np.int64)
        anchor_index = sample_ids + self.seq_len - 1

        anchors = self.raw_data[anchor_index]  # [B, C]
        recovered = anchors[:, None, :] + np.cumsum(diff_raw, axis=1)

        return recovered.astype(np.float32)

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