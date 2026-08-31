# Author  : mkw
# Desc    : Abilene dataset loader with missing-aware first-order diff + second diff + raw input features
# Shape   : raw data [T, C] = [3000, 144]
# X 特征拼接版本:
#   [一阶差分标准化, 一阶差分掩码, 二阶差分原始值, 二阶差分掩码, 原始值, 原始值掩码]

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

        # 这里沿用原来的返回接口，把 x_mark 用作未来标签有效掩码。
        # 形状为 [pred_len, target_dim]。
        # 模型 forward 当前不会使用 x_mark，训练脚本会用它做 masked loss。
        x_mark = y_mask.astype(np.float32)

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
    """Abilene multivariate time-series dataset with missing-aware trend features.

    X feature order on the last dimension:
        [first-order diff normalized,
         first-order diff valid mask,
         second-order diff raw,
         second-order diff valid mask,
         raw value placeholder,
         raw value valid mask]

    For raw data [T, C], model input X becomes [N, seq_len, 6 * C].
    Label Y remains first-order diff normalized [N, pred_len, 1], so the
    original recover_level_from_diff() evaluation path is unchanged.

    Missing convention:
        By default, zero or non-finite values are treated as missing.
        If zero is a valid traffic value in your data, set
        config.mask_zero_as_missing = False.
    """

    def __init__(self, config, trainX=None):
        del trainX

        self.config = config
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.stride = int(getattr(self.config, "stride", 1))
        self.use_missing_aware_encoding = bool(
            getattr(self.config, "use_missing_aware_encoding", True)
        )
        setattr(
            self.config,
            "use_missing_aware_encoding",
            self.use_missing_aware_encoding,
        )

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

        self.target_col = int(getattr(self.config, "target_col", 0))
        self.target_dim = 1
        self.num_vars = None
        self.input_feature_dim = None

        setattr(self.config, "target_col", self.target_col)
        setattr(self.config, "target_dim", self.target_dim)

        # 原始数据占位值，评估阶段累加还原会用到。
        # 缺失位置被置为 0，但是否有效由 raw_mask 标记。
        self.raw_data = None
        self.raw_mask = None
        # Input-only mask after optional artificial corruption. Exposed for
        # reproducibility checks; labels continue to use ``raw_mask``.
        self.input_mask = None

        # 差分归一化参数，形状都是 [C]
        self.mean = None
        self.std = None
        self.mini = 0.0

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

        if np.isinf(data).any():
            raise ValueError("Inf found in csv.")

        # 直接从数据本身推断变量数
        self.num_vars = int(data.shape[1])

        # X 会在最后一维拼接六类特征：
        #   1) 一阶差分标准化值
        #   2) 一阶差分有效掩码
        #   3) 二阶差分原始值
        #   4) 二阶差分有效掩码
        #   5) 原始值占位
        #   6) 原始值有效掩码
        # 因此模型实际输入维度是 6 * C。
        if self.use_missing_aware_encoding:
            self.input_feature_dim = self.num_vars * 6
            self.input_feature_order = (
                "[diff_norm, diff_mask, second_diff_raw, "
                "second_diff_mask, raw, raw_mask]"
            )
            missing_aware_groups = 6
        else:
            self.input_feature_dim = self.num_vars * 3
            self.input_feature_order = "[diff_norm, second_diff_raw, raw]"
            missing_aware_groups = 0

        # 回写 config，给后面的模型初始化用。
        # num_vars 保留原始变量数 C；enc_in 表示模型实际输入维度 6C。
        setattr(self.config, "num_vars", self.num_vars)
        setattr(self.config, "input_feature_dim", self.input_feature_dim)
        setattr(self.config, "enc_in", self.input_feature_dim)
        setattr(self.config, "missing_aware_groups", missing_aware_groups)

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

    def _build_observation_mask(self, data: np.ndarray) -> np.ndarray:
        """构造原始观测掩码 M。

        m(i,n)=1 表示 x(i,n) 是真实观测；
        m(i,n)=0 表示该位置缺失。

        默认将 0 和 NaN 都视为缺失；如果 0 是合法真实值，
        请设置 config.mask_zero_as_missing=False。
        """
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
        """Randomly remove observed entries from the input observation mask.

        This mask is used only to build model inputs. Label masks are kept from
        the original observation mask, so artificial input missingness does not
        change supervised-loss or evaluation positions.
        """
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
            target_candidate = np.zeros_like(candidate)
            target_candidate[:, self.target_col] = candidate[:, self.target_col]
            candidate = target_candidate

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

        removed_flat = rng.choice(
            candidate_idx,
            size=remove_count,
            replace=False,
        )
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

    def _apply_block_artificial_missing(
        self,
        mask: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> np.ndarray:
        """Remove contiguous time-by-variable blocks from model inputs.

        ``artificial_missing_rate`` is the target fraction of originally
        observed entries removed independently inside each selected split.
        Every sampled block spans ``artificial_missing_block_length`` adjacent
        time steps and ``artificial_missing_column_rate`` of the variables.
        Artificial corruption never changes the label mask.
        """
        rate = float(getattr(self.config, "artificial_missing_rate", 0.0))
        if rate <= 0.0:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)
        if rate >= 1.0:
            raise ValueError(
                "artificial_missing_rate must be in [0, 1). "
                f"Got {rate}."
            )

        block_length = int(
            getattr(self.config, "artificial_missing_block_length", 12)
        )
        column_rate = float(
            getattr(self.config, "artificial_missing_column_rate", 1.0)
        )
        if block_length < 1:
            raise ValueError(
                "artificial_missing_block_length must be >= 1. "
                f"Got {block_length}."
            )
        if not 0.0 < column_rate <= 1.0:
            raise ValueError(
                "artificial_missing_column_rate must be in (0, 1]. "
                f"Got {column_rate}."
            )

        requested_splits = self._get_artificial_missing_splits()
        if not requested_splits:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)

        aliases = {
            "train": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
        }
        if "all" in requested_splits:
            selected_splits = {"train", "val", "test"}
        else:
            unknown = requested_splits.difference(aliases)
            if unknown:
                raise ValueError(
                    "Unknown artificial_missing_splits item(s) "
                    f"{sorted(unknown)}. Use train,val,test or all."
                )
            selected_splits = {aliases[name] for name in requested_splits}

        split_ranges = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, mask.shape[0]),
        }
        seed = int(getattr(self.config, "artificial_missing_seed", 2026))
        target_only = bool(
            getattr(self.config, "artificial_missing_target_only", False)
        )
        rng = np.random.default_rng(seed)
        new_mask = mask.copy().astype(np.float32)
        eligible_global = np.zeros_like(mask, dtype=bool)
        split_stats = []

        for split_name in ("train", "val", "test"):
            if split_name not in selected_splits:
                continue

            split_start, split_end = split_ranges[split_name]
            split_length = split_end - split_start
            if split_length < block_length:
                raise ValueError(
                    f"Split '{split_name}' has length {split_length}, shorter "
                    f"than artificial_missing_block_length={block_length}."
                )

            eligible = np.zeros_like(mask, dtype=bool)
            if target_only:
                eligible[split_start:split_end, self.target_col] = True
                selectable_columns = np.asarray([self.target_col], dtype=np.int64)
                columns_per_block = 1
            else:
                eligible[split_start:split_end, :] = True
                selectable_columns = np.arange(mask.shape[1], dtype=np.int64)
                columns_per_block = max(
                    1,
                    int(round(mask.shape[1] * column_rate)),
                )

            eligible &= mask > 0.5
            eligible_global |= eligible
            total_candidates = int(eligible.sum())
            target_remove = int(round(total_candidates * rate))
            if total_candidates == 0 or target_remove == 0:
                split_stats.append((split_name, 0, total_candidates, 0))
                continue

            removed_count = 0
            block_count = 0
            # Overlap and pre-existing missing values can make a sampled block
            # add no new removals. The generous bound prevents an accidental
            # infinite loop while retaining deterministic masks.
            estimated_blocks = int(
                np.ceil(
                    target_remove
                    / max(block_length * columns_per_block, 1)
                )
            )
            max_attempts = max(1000, estimated_blocks * 50)
            attempts = 0

            while removed_count < target_remove and attempts < max_attempts:
                attempts += 1
                block_start = int(
                    rng.integers(
                        split_start,
                        split_end - block_length + 1,
                    )
                )
                block_end = block_start + block_length
                columns = rng.choice(
                    selectable_columns,
                    size=min(columns_per_block, selectable_columns.size),
                    replace=False,
                )

                rows = np.arange(block_start, block_end, dtype=np.int64)
                block_index = np.ix_(rows, columns)
                before = new_mask[block_index] > 0.5
                allowed = eligible[block_index]
                newly_removed = before & allowed
                added = int(newly_removed.sum())
                if added == 0:
                    continue

                block_values = new_mask[block_index]
                block_values[allowed] = 0.0
                new_mask[block_index] = block_values
                removed_count += added
                block_count += 1

            if removed_count < target_remove:
                raise RuntimeError(
                    f"Unable to reach block-missing target in split "
                    f"'{split_name}': removed={removed_count}, "
                    f"target={target_remove}, attempts={attempts}."
                )

            split_stats.append(
                (split_name, removed_count, total_candidates, block_count)
            )

        removed_global = eligible_global & (new_mask <= 0.5)
        total_global = int(eligible_global.sum())
        actual_rate = (
            float(removed_global.sum()) / total_global
            if total_global > 0
            else 0.0
        )
        setattr(self.config, "artificial_missing_actual_rate", actual_rate)

        split_text = "; ".join(
            f"{name}:removed={removed}/{total},blocks={blocks}"
            for name, removed, total, blocks in split_stats
        )
        print(
            "[Block Missing] "
            f"target_rate={rate:.4f}, actual_rate={actual_rate:.4f}, "
            f"block_length={block_length}, column_rate={column_rate:.4f}, "
            f"target_only={target_only}, seed={seed}, {split_text}"
        )

        return new_mask.astype(np.float32)

    def _apply_artificial_missing(
        self,
        mask: np.ndarray,
        train_end: int,
        val_end: int,
    ) -> np.ndarray:
        """Dispatch the configured input-only artificial missing pattern."""
        rate = float(getattr(self.config, "artificial_missing_rate", 0.0))
        if rate <= 0.0:
            setattr(self.config, "artificial_missing_actual_rate", 0.0)
            return mask.astype(np.float32)

        pattern = str(
            getattr(self.config, "artificial_missing_pattern", "random_point")
        ).strip().lower().replace("-", "_")
        if pattern in {"random", "random_point", "point"}:
            return self._apply_random_artificial_missing(mask, train_end, val_end)
        if pattern in {"block", "time_block", "structured_block"}:
            return self._apply_block_artificial_missing(mask, train_end, val_end)
        raise ValueError(
            "artificial_missing_pattern must be random_point or time_block. "
            f"Got {pattern!r}."
        )

    @staticmethod
    def _first_order_diff(
        data: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Missing-aware first-order difference.

        一阶差分有效掩码：
            m_delta[t] = m[t] * m[t-1]

        只有相邻两个时间片均真实观测时，才计算：
            diff[t] = data[t] - data[t-1]

        其他位置置为 0 占位。
        """
        diff = np.zeros_like(data, dtype=np.float32)
        diff_mask = np.zeros_like(mask, dtype=np.float32)

        valid = (mask[1:, :] > 0.5) & (mask[:-1, :] > 0.5)
        diff_mask[1:, :] = valid.astype(np.float32)

        raw_diff = data[1:, :] - data[:-1, :]
        diff[1:, :] = np.where(valid, raw_diff, 0.0)

        return diff.astype(np.float32), diff_mask.astype(np.float32)

    @staticmethod
    def _second_order_diff(
        data: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Missing-aware second-order difference.

        二阶差分有效掩码：
            m_delta2[t] = m[t] * m[t-1] * m[t-2]

        只有连续三个时间片均真实观测时，才计算：
            second_diff[t] = data[t] - 2 * data[t-1] + data[t-2]

        其他位置置为 0 占位。
        """
        second_diff = np.zeros_like(data, dtype=np.float32)
        second_mask = np.zeros_like(mask, dtype=np.float32)

        valid = (
            (mask[2:, :] > 0.5)
            & (mask[1:-1, :] > 0.5)
            & (mask[:-2, :] > 0.5)
        )
        second_mask[2:, :] = valid.astype(np.float32)

        raw_second = data[2:, :] - 2.0 * data[1:-1, :] + data[:-2, :]
        second_diff[2:, :] = np.where(valid, raw_second, 0.0)

        return second_diff.astype(np.float32), second_mask.astype(np.float32)

    @classmethod
    def _diff_std_normalization(
        cls,
        data: np.ndarray,
        mask: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Missing-aware first-order difference followed by z-score normalization.

        data: [T, C]
        mask: [T, C]

        return:
            norm:      [T, C] normalized first-order diff, invalid entries are 0
            diff_mask: [T, C] valid mask of first-order diff
            mean:      [C]
            std:       [C]
            mini: kept for compatibility with the old dataset API
        """
        diff, diff_mask = cls._first_order_diff(data, mask)

        if mean is None or std is None:
            valid = diff_mask > 0.5
            count = valid.sum(axis=0).astype(np.float32)

            safe_count = np.maximum(count, 1.0)
            mean_est = (diff * valid).sum(axis=0) / safe_count

            centered = np.where(valid, diff - mean_est.reshape(1, -1), 0.0)
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

        norm = np.zeros_like(diff, dtype=np.float32)
        valid = diff_mask > 0.5
        norm_valid = (diff - mean.reshape(1, -1)) / std.reshape(1, -1)
        norm = np.where(valid, norm_valid, 0.0).astype(np.float32)

        mini = 0.0

        return (
            norm.astype(np.float32),
            diff_mask.astype(np.float32),
            mean.astype(np.float32),
            std.astype(np.float32),
            mini,
        )

    def _fit_diff_normalizer(self, train_raw: np.ndarray, train_mask: np.ndarray):
        """
        只用训练集有效一阶差分拟合归一化参数，避免验证/测试泄漏。
        """
        _, _, self.mean, self.std, self.mini = self._diff_std_normalization(
            train_raw,
            train_mask,
        )

    def _transform_diff(
        self,
        raw: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练集 mean/std，把原始序列转成标准化一阶差分序列。
        无效差分位置保持为 0 占位。
        """
        norm, diff_mask, _, _, _ = self._diff_std_normalization(
            raw,
            mask,
            mean=self.mean,
            std=self.std,
        )
        return norm, diff_mask

    def _make_windows(
        self,
        diff_norm: np.ndarray,
        diff_mask: np.ndarray,
        second_diff_raw: np.ndarray,
        second_diff_mask: np.ndarray,
        raw: np.ndarray,
        raw_mask: np.ndarray,
        label_diff_norm: np.ndarray | None = None,
        label_diff_mask: np.ndarray | None = None,
    ):
        """
        构造窗口。

        diff_norm:        [T, C] 一阶差分标准化值
        diff_mask:        [T, C] 一阶差分有效掩码
        second_diff_raw:  [T, C] 二阶差分原始值，不做标准化
        second_diff_mask: [T, C] 二阶差分有效掩码
        raw:              [T, C] 原始值占位，不做标准化
        raw_mask:         [T, C] 原始值有效掩码

        return:
            x: [N, seq_len, 6 * C]
               最后一维拼接顺序为：
               [一阶差分标准化, 一阶差分掩码,
                二阶差分原始值, 二阶差分掩码,
                原始值, 原始值掩码]
            y: [N, pred_len, 1]
               仍然是一阶差分标准化后的目标列。
            y_mask: [N, pred_len, 1]
               未来一阶差分标签有效掩码，用于 masked loss / evaluation。
        """
        if label_diff_norm is None:
            label_diff_norm = diff_norm
        if label_diff_mask is None:
            label_diff_mask = diff_mask

        shapes = {
            "diff_norm": diff_norm.shape,
            "diff_mask": diff_mask.shape,
            "second_diff_raw": second_diff_raw.shape,
            "second_diff_mask": second_diff_mask.shape,
            "raw": raw.shape,
            "raw_mask": raw_mask.shape,
            "label_diff_norm": label_diff_norm.shape,
            "label_diff_mask": label_diff_mask.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"All input arrays must have same shape, got {shapes}")

        total = diff_norm.shape[0] - self.seq_len - self.pred_len + 1

        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={diff_norm.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []
        y_mask_list = []

        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len

            x_diff_norm = diff_norm[start : start + self.seq_len, :]
            x_diff_mask = diff_mask[start : start + self.seq_len, :]

            x_second_diff_raw = second_diff_raw[start : start + self.seq_len, :]
            x_second_diff_mask = second_diff_mask[start : start + self.seq_len, :]

            x_raw = raw[start : start + self.seq_len, :]
            x_raw_mask = raw_mask[start : start + self.seq_len, :]

            if self.use_missing_aware_encoding:
                x = np.concatenate(
                    [
                        x_diff_norm,
                        x_diff_mask,
                        x_second_diff_raw,
                        x_second_diff_mask,
                        x_raw,
                        x_raw_mask,
                    ],
                    axis=-1,
                )
            else:
                x = np.concatenate(
                    [
                        x_diff_norm,
                        x_second_diff_raw,
                        x_raw,
                    ],
                    axis=-1,
                )

            # y 不拼接，仍然只预测目标列的一阶差分标准化值。
            y = label_diff_norm[y_start:y_end, self.target_col : self.target_col + 1]
            y_mask = label_diff_mask[y_start:y_end, self.target_col : self.target_col + 1]

            x_list.append(x)
            y_list.append(y)
            y_mask_list.append(y_mask)

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        y_mask = np.stack(y_mask_list, axis=0).astype(np.float32)

        return x, y, y_mask

    def _load_and_build(self):
        raw_loaded = self._load_csv()

        full_mask = self._build_observation_mask(raw_loaded)
        split_sizes, (train_end, val_end) = self._split_ratio(len(raw_loaded))
        input_mask = self._apply_artificial_missing(
            full_mask,
            train_end,
            val_end,
        )
        data = np.where(full_mask > 0.5, raw_loaded, 0.0).astype(np.float32)
        input_data = np.where(input_mask > 0.5, raw_loaded, 0.0).astype(np.float32)

        self.raw_data = data
        self.raw_mask = full_mask.astype(np.float32)
        self.input_mask = input_mask.astype(np.float32)

        train_raw = data[:train_end]
        train_mask = full_mask[:train_end]
        train_input_raw = input_data[:train_end]
        train_input_mask = input_mask[:train_end]

        # 验证集和测试集保留前 seq_len 个历史点作为输入上下文
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

        # 只用训练集有效一阶差分拟合归一化参数
        self._fit_diff_normalizer(train_raw, train_mask)

        train_norm, train_diff_mask = self._transform_diff(train_raw, train_mask)
        val_norm, val_diff_mask = self._transform_diff(val_raw, val_mask)
        test_norm, test_diff_mask = self._transform_diff(test_raw, test_mask)
        train_input_norm, train_input_diff_mask = self._transform_diff(
            train_input_raw,
            train_input_mask,
        )
        val_input_norm, val_input_diff_mask = self._transform_diff(
            val_input_raw,
            val_input_mask,
        )
        test_input_norm, test_input_diff_mask = self._transform_diff(
            test_input_raw,
            test_input_mask,
        )

        # 二阶差分原始值，不做标准化。
        train_second_diff, train_second_mask = self._second_order_diff(
            train_input_raw,
            train_input_mask,
        )
        val_second_diff, val_second_mask = self._second_order_diff(
            val_input_raw,
            val_input_mask,
        )
        test_second_diff, test_second_mask = self._second_order_diff(
            test_input_raw,
            test_input_mask,
        )

        x_train, y_train, y_train_mask = self._make_windows(
            train_input_norm,
            train_input_diff_mask,
            train_second_diff,
            train_second_mask,
            train_input_raw,
            train_input_mask,
            label_diff_norm=train_norm,
            label_diff_mask=train_diff_mask,
        )
        x_val, y_val, y_val_mask = self._make_windows(
            val_input_norm,
            val_input_diff_mask,
            val_second_diff,
            val_second_mask,
            val_input_raw,
            val_input_mask,
            label_diff_norm=val_norm,
            label_diff_mask=val_diff_mask,
        )
        x_test, y_test, y_test_mask = self._make_windows(
            test_input_norm,
            test_input_diff_mask,
            test_second_diff,
            test_second_mask,
            test_input_raw,
            test_input_mask,
            label_diff_norm=test_norm,
            label_diff_mask=test_diff_mask,
        )

        # ===== 计算训练集点级 |diff| q90（用于 Tail 指标） =====
        # 只使用有效未来一阶差分标签，避免缺失占位 0 污染分位数。
        train_diff_raw = self.inverse_diff_norm(y_train)
        valid_train_diff = y_train_mask > 0.5
        all_diff_vals = np.abs(train_diff_raw[valid_train_diff])
        all_diff_vals = all_diff_vals[np.isfinite(all_diff_vals)]

        if len(all_diff_vals) > 0:
            self.tail_q90 = float(np.quantile(all_diff_vals, 0.90))
        else:
            self.tail_q90 = 0.0

        setattr(self.config, "tail_q90", self.tail_q90)
        print(f"[Tail Threshold] |diff| q90={self.tail_q90:.6f}")

        # ===== 计算训练集点级原始值 q90/q99（用于 level 指标） =====
        # 只使用有效原始未来标签。
        raw_y_list = []
        raw_y_mask_list = []
        total = train_raw.shape[0] - self.seq_len - self.pred_len + 1

        for start in range(0, total, self.stride):
            y_start = start + self.seq_len
            y_end = y_start + self.pred_len
            raw_y = train_raw[y_start:y_end, self.target_col : self.target_col + 1]
            raw_y_mask = train_mask[y_start:y_end, self.target_col : self.target_col + 1]
            raw_y_list.append(raw_y)
            raw_y_mask_list.append(raw_y_mask)

        if len(raw_y_list) > 0:
            all_raw_vals = np.stack(raw_y_list, axis=0).astype(np.float32)
            all_raw_masks = np.stack(raw_y_mask_list, axis=0).astype(np.float32)

            all_raw_vals = all_raw_vals[all_raw_masks > 0.5]
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
            f"x_train={x_train.shape}, y_train={y_train.shape}, mask_train={y_train_mask.shape}",
            f"x_val={x_val.shape}, y_val={y_val.shape}, mask_val={y_val_mask.shape}",
            f"x_test={x_test.shape}, y_test={y_test.shape}, mask_test={y_test_mask.shape}",
        )
        print(
            f"[Dataset] input feature concat: "
            f"num_vars={self.num_vars}, enc_in={self.input_feature_dim}, "
            f"use_missing_aware_encoding={self.use_missing_aware_encoding}, "
            f"feature_order={self.input_feature_order}"
        )
        print(
            f"[Dataset] missing convention: mask_zero_as_missing={self.mask_zero_as_missing}"
        )
        print(
            f"[Dataset] many-to-one target: "
            f"target_col={self.target_col}, target_dim={self.target_dim}"
        )
        print(
            f"[Dataset] diff norm mean/std shape: "
            f"mean={self.mean.shape}, std={self.std.shape}"
        )

    def inverse_diff_norm(self, diff_norm: np.ndarray) -> np.ndarray:
        """
        把标准化差分还原成原始差分。

        diff_norm: [..., C] or [..., 1]
        return:    [..., C] or [..., 1]
        """
        diff_norm = np.asarray(diff_norm)

        if diff_norm.shape[-1] == self.target_dim:
            mean = self.mean[self.target_col : self.target_col + 1]
            std = self.std[self.target_col : self.target_col + 1]
        else:
            mean = self.mean
            std = self.std

        return diff_norm * std + mean

    @staticmethod
    def recover_level_valid_mask_from_diff_mask(diff_mask: np.ndarray) -> np.ndarray:
        """
        从未来一阶差分有效掩码恢复原始值空间的评估掩码。

        因为第 h 步原始值由前 1..h 个差分累加得到，所以只有当从
        anchor 到当前步之间的所有差分均有效时，该原始值预测才参与评估。
        """
        diff_mask = np.asarray(diff_mask, dtype=np.float32)
        return np.cumprod(diff_mask, axis=1).astype(np.float32)

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

        if diff_norm.shape[-1] == self.target_dim:
            anchors = self.raw_data[
                anchor_index,
                self.target_col : self.target_col + 1,
            ]  # [B, 1]
        else:
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

    def get_raw_mask(self):
        return self.raw_mask
