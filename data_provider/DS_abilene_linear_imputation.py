"""Two-stage forecasting dataset with window-local linear interpolation.

This module is intentionally separate from ``DS_abilene_single_mask``.  The
existing loader keeps missing inputs as zero-valued placeholders; this loader
uses the same artificial-missing generators, but imputes every historical
window before it is passed to a forecasting baseline.

Artificial missingness affects inputs only.  Forecast labels and label masks
always come from the uncorrupted series.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from data_provider.DS_abilene_diff_single_mask import DS as _DATPMissingDS
from data_provider.DS_abilene_single_mask import (
    DS as _ZeroPlaceholderDS,
    TimeSeriesDataset,
)


class DS(_ZeroPlaceholderDS):
    """Baseline dataset implementing ``missing -> linear interpolation``."""

    # Reuse the exact deterministic mask generator used by DATP-Net so the
    # direct and two-stage experiments can share seeds and missing positions.
    _apply_block_artificial_missing = _DATPMissingDS._apply_block_artificial_missing
    _apply_artificial_missing = _DATPMissingDS._apply_artificial_missing

    def __init__(self, config, trainX=None):
        setattr(config, "input_imputation", "linear")
        setattr(config, "two_stage_forecasting", True)
        self.input_mask = None
        self.interpolation_stats = {
            "windows": 0,
            "missing_values": 0,
            "all_missing_columns": 0,
        }
        super().__init__(config, trainX=trainX)

    @staticmethod
    def _linear_interpolate_window(
        raw_window: np.ndarray,
        mask_window: np.ndarray,
        fallback_mean: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        """Interpolate one [seq_len, variables] history without future access.

        ``numpy.interp`` performs linear interpolation for internal gaps and
        nearest-value filling at the left/right boundaries.  A column with no
        observation in the window falls back to that variable's training mean.
        """

        raw_window = np.asarray(raw_window, dtype=np.float32)
        mask_window = np.asarray(mask_window) > 0.5
        fallback_mean = np.asarray(fallback_mean, dtype=np.float32).reshape(-1)

        if raw_window.ndim != 2 or raw_window.shape != mask_window.shape:
            raise ValueError(
                "raw_window and mask_window must be equal 2D arrays, got "
                f"{raw_window.shape} and {mask_window.shape}"
            )
        if raw_window.shape[1] != fallback_mean.size:
            raise ValueError(
                f"fallback_mean has {fallback_mean.size} values for "
                f"{raw_window.shape[1]} variables"
            )

        result = raw_window.copy()
        time_index = np.arange(raw_window.shape[0], dtype=np.float64)
        missing_count = 0
        all_missing_columns = 0

        for column in range(raw_window.shape[1]):
            valid = mask_window[:, column] & np.isfinite(raw_window[:, column])
            missing = ~valid
            missing_count += int(missing.sum())

            if valid.any():
                observed_index = time_index[valid]
                observed_values = raw_window[valid, column].astype(np.float64)
                result[:, column] = np.interp(
                    time_index,
                    observed_index,
                    observed_values,
                ).astype(np.float32)
            else:
                result[:, column] = fallback_mean[column]
                all_missing_columns += 1

        if not np.isfinite(result).all():
            raise ValueError("Linear interpolation produced NaN or Inf values")

        return result.astype(np.float32), missing_count, all_missing_columns

    def _make_linear_windows(
        self,
        input_raw: np.ndarray,
        input_mask: np.ndarray,
        label_norm_data: np.ndarray,
        label_mask: np.ndarray,
    ):
        shapes = {
            input_raw.shape,
            input_mask.shape,
            label_norm_data.shape,
            label_mask.shape,
        }
        if len(shapes) != 1:
            raise ValueError(
                "input_raw, input_mask, label_norm_data and label_mask must "
                "have identical shapes"
            )

        total = input_raw.shape[0] - self.seq_len - self.pred_len + 1
        if total <= 0:
            raise ValueError(
                f"Sequence too short: T={input_raw.shape[0]}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

        x_list = []
        y_list = []
        y_mask_list = []
        missing_values = 0
        all_missing_columns = 0

        for start in range(0, total, self.stride):
            input_end = start + self.seq_len
            label_end = input_end + self.pred_len

            imputed_raw, window_missing, window_all_missing = (
                self._linear_interpolate_window(
                    input_raw[start:input_end, :],
                    input_mask[start:input_end, :],
                    self.mean,
                )
            )
            input_norm = (
                (imputed_raw - self.mean.reshape(1, -1))
                / self.std.reshape(1, -1)
            ).astype(np.float32)

            x_list.append(input_norm)
            y_list.append(
                label_norm_data[
                    input_end:label_end,
                    self.target_col : self.target_col + 1,
                ]
            )
            y_mask_list.append(
                label_mask[
                    input_end:label_end,
                    self.target_col : self.target_col + 1,
                ]
            )
            missing_values += window_missing
            all_missing_columns += window_all_missing

        self.interpolation_stats["windows"] += len(x_list)
        self.interpolation_stats["missing_values"] += missing_values
        self.interpolation_stats["all_missing_columns"] += all_missing_columns

        x = np.stack(x_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)
        y_mask = np.stack(y_mask_list, axis=0).astype(np.float32)

        if not np.isfinite(x).all():
            raise ValueError("Non-finite baseline input remains after interpolation")

        return x, y, y_mask

    def _load_and_build(self):
        raw_loaded = self._load_csv()
        _, (train_end, val_end) = self._split_ratio(len(raw_loaded))

        full_mask = self._build_observation_mask(raw_loaded)
        input_mask = self._apply_artificial_missing(
            full_mask,
            train_end,
            val_end,
        )

        # Labels retain the complete/original observation state.  The hidden
        # input values are not copied into model inputs; interpolation only sees
        # the values whose input mask is one.
        label_raw = np.where(full_mask > 0.5, raw_loaded, 0.0).astype(np.float32)
        input_raw = np.where(input_mask > 0.5, raw_loaded, np.nan).astype(np.float32)
        self.raw_data = label_raw
        self.raw_mask = full_mask.astype(np.float32)
        self.input_mask = input_mask.astype(np.float32)

        train_raw = label_raw[:train_end]
        train_mask = full_mask[:train_end]
        train_input_raw = input_raw[:train_end]
        train_input_mask = input_mask[:train_end]

        val_start = max(0, train_end - self.seq_len)
        test_start = max(0, val_end - self.seq_len)

        val_raw = label_raw[val_start:val_end]
        val_mask = full_mask[val_start:val_end]
        val_input_raw = input_raw[val_start:val_end]
        val_input_mask = input_mask[val_start:val_end]

        test_raw = label_raw[test_start:]
        test_mask = full_mask[test_start:]
        test_input_raw = input_raw[test_start:]
        test_input_mask = input_mask[test_start:]

        # Keep the normalization convention aligned with the direct DATP-Net
        # experiment: statistics are fitted on the original training labels.
        self._fit_normalizer(train_raw, train_mask)
        train_norm = self._transform_raw_values(train_raw, train_mask)
        val_norm = self._transform_raw_values(val_raw, val_mask)
        test_norm = self._transform_raw_values(test_raw, test_mask)

        x_train, y_train, y_train_mask = self._make_linear_windows(
            train_input_raw,
            train_input_mask,
            train_norm,
            train_mask,
        )
        x_val, y_val, y_val_mask = self._make_linear_windows(
            val_input_raw,
            val_input_mask,
            val_norm,
            val_mask,
        )
        x_test, y_test, y_test_mask = self._make_linear_windows(
            test_input_raw,
            test_input_mask,
            test_norm,
            test_mask,
        )

        train_value_raw = self.inverse_value_norm(y_train)
        observed_train_values = np.abs(
            train_value_raw[y_train_mask > 0.5]
        ).reshape(-1)
        observed_train_values = observed_train_values[
            np.isfinite(observed_train_values)
        ]
        self.tail_q90 = (
            float(np.quantile(observed_train_values, 0.90))
            if observed_train_values.size
            else 0.0
        )
        self.raw_q90 = self.tail_q90
        self.raw_q99 = (
            float(np.quantile(observed_train_values, 0.99))
            if observed_train_values.size
            else 0.0
        )
        setattr(self.config, "tail_q90", self.tail_q90)
        setattr(self.config, "raw_q90", self.raw_q90)
        setattr(self.config, "raw_q99", self.raw_q99)
        setattr(
            self.config,
            "linear_interpolation_all_missing_columns",
            self.interpolation_stats["all_missing_columns"],
        )

        batch_size = int(self.config.bs)
        num_workers = int(getattr(self.config, "num_workers", 0))
        common = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        self.train_data_loader = DataLoader(
            TimeSeriesDataset(
                x_train,
                y_train,
                y_train_mask,
                self.config,
                id_offset=0,
                stride=self.stride,
            ),
            shuffle=True,
            **common,
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
            shuffle=False,
            **common,
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
            shuffle=False,
            **common,
        )

        print(
            "[Two-Stage Linear] "
            f"dataset={self.dataset}, pattern="
            f"{getattr(self.config, 'artificial_missing_pattern', 'random_point')}, "
            f"target_rate={getattr(self.config, 'artificial_missing_rate', 0.0):.4f}, "
            f"actual_rate={getattr(self.config, 'artificial_missing_actual_rate', 0.0):.4f}"
        )
        print(
            "[Two-Stage Linear] window-local interpolation: "
            f"windows={self.interpolation_stats['windows']}, "
            f"filled_values={self.interpolation_stats['missing_values']}, "
            f"all_missing_column_fallbacks="
            f"{self.interpolation_stats['all_missing_columns']}"
        )
        print(
            "[Two-Stage Linear] windows: "
            f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}; "
            "labels remain uncorrupted"
        )

    def get_input_mask(self):
        return self.input_mask

