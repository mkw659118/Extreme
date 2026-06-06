import torch as t
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def _apply_metric_valid_mask(real_values, pred_values, valid_mask=None, config=None):
    """
    指标计算用的有效位置筛选。

    优先使用显式 valid_mask：
        valid_mask == 1 的位置参与计算；
        valid_mask == 0 的位置不参与计算。

    如果没有传入 valid_mask，则至少要求 real/pred 都是有限值。
    对旧流程兼容：当 config.mask_zero_as_missing=True 时，real==0
    会被视为缺失标签并排除；不再因为 pred==0 而排除位置。
    """
    real_values = np.asarray(real_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    if real_values.shape != pred_values.shape:
        raise ValueError(
            f"realVec and estiVec must have same shape, "
            f"got {real_values.shape} vs {pred_values.shape}"
        )

    valid = np.isfinite(real_values) & np.isfinite(pred_values)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != real_values.shape:
            if valid_mask.ndim == real_values.ndim - 1:
                valid_mask = np.expand_dims(valid_mask, axis=-1)
            if valid_mask.shape[-1] == 1 and real_values.shape[-1] > 1:
                valid_mask = np.broadcast_to(valid_mask, real_values.shape)
            if valid_mask.shape != real_values.shape:
                raise ValueError(
                    f"valid_mask shape {valid_mask.shape} does not match "
                    f"real/pred shape {real_values.shape}"
                )
        valid = valid & (valid_mask > 0.5)
    else:
        mask_zero = bool(getattr(config, "mask_zero_as_missing", False)) if config is not None else False
        if mask_zero:
            valid = valid & (real_values != 0)

    valid_count = int(np.sum(valid))
    if valid_count == 0:
        raise ValueError(
            "No valid entries for metric computation: "
            "all positions are masked or non-finite."
        )

    return real_values[valid], pred_values[valid], valid, valid_count


# Backward-compatible alias. The semantics are updated: prediction value 0 is
# no longer treated as invalid; only missing-label positions should be masked.
def _apply_nonzero_valid_mask(real_values, pred_values):
    return _apply_metric_valid_mask(real_values, pred_values, valid_mask=None, config=None)


def ErrorMetrics(realVec, estiVec, config, mode, valid_mask=None):
    """根据任务类型选择合适的误差计算方式。"""
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)

    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    if isinstance(valid_mask, t.Tensor):
        valid_mask = valid_mask.cpu().detach().numpy()

    return compute_regression_metrics(realVec, estiVec, config, mode, valid_mask=valid_mask)
    # return compute_regression_metrics_rolling(realVec, estiVec, config, config.pred_len, valid_mask=valid_mask)


def compute_regression_metrics_rolling(realVec, estiVec, config, rm, valid_mask=None):
    """计算回归任务的误差指标，支持滚动窗口评估。"""
    realVec = np.array(realVec)
    estiVec = np.array(estiVec)
    valid_mask = None if valid_mask is None else np.asarray(valid_mask)

    ll = int(len(estiVec) / config.pred_len)

    esti_all = []
    real_all = []
    mask_all = []

    for i in range(ll):
        esti_window = estiVec[i * config.pred_len: (i * config.pred_len + rm)]
        real_window = realVec[i * config.pred_len: (i * config.pred_len + rm)]
        esti_all.extend(esti_window)
        real_all.extend(real_window)

        if valid_mask is not None:
            mask_window = valid_mask[i * config.pred_len: (i * config.pred_len + rm)]
            mask_all.extend(mask_window)

    esti_all = np.array(esti_all)
    real_all = np.array(real_all)
    mask_all = None if valid_mask is None else np.array(mask_all)

    real_all, esti_all, _, valid_count = _apply_metric_valid_mask(
        real_all,
        esti_all,
        valid_mask=mask_all,
        config=config,
    )

    eps = 1e-8
    absError = np.abs(esti_all - real_all)

    MAE = np.mean(np.abs(real_all - esti_all))
    MSE = np.mean((real_all - esti_all) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((real_all - esti_all) / (np.abs(real_all) + eps)))

    NMAE = np.sum(absError) / (np.sum(np.abs(real_all)) + eps)
    NRMSE = np.sqrt(np.sum((real_all - esti_all) ** 2)) / (
        np.sqrt(np.sum(real_all ** 2)) + eps
    )

    thresholds = [0.01, 0.05, 0.10]
    Acc = [
        np.mean((absError < (np.abs(real_all) * threshold)).astype(float))
        for threshold in thresholds
    ]

    all_metrics = {
        "MAE_8": MAE,
        "MSE_8": MSE,
        "RMSE_8": RMSE,
        "MAPE_8": MAPE,
        "NMAE_8": NMAE,
        "NRMSE_8": NRMSE,
        "Acc_10%_8": Acc[2],
        "Valid_count_8": valid_count,
    }

    return all_metrics


def _mean_cosine_similarity(y_true, y_pred, eps_val=1e-8):
    """
    计算预测序列和真实序列的中心化余弦相似度。
    单目标预测时，将所有点拉平成一条完整曲线后计算 COS。
    多输出预测时，沿最后一个维度计算 per-sample COS 后取平均。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, "
            f"got {y_true.shape} vs {y_pred.shape}"
        )

    if y_true.size == 0:
        return np.nan

    if y_true.ndim == 1:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)

        y_true = y_true - np.mean(y_true)
        y_pred = y_pred - np.mean(y_pred)

        dot = float(np.sum(y_true * y_pred))
        norm_true = float(np.linalg.norm(y_true))
        norm_pred = float(np.linalg.norm(y_pred))
        return float(dot / (norm_true * norm_pred + eps_val))

    if (y_true.ndim == 2 and y_true.shape[-1] == 1) or (
        y_true.ndim == 3 and y_true.shape[-1] == 1
    ):
        y_true = y_true.reshape(-1).astype(float)
        y_pred = y_pred.reshape(-1).astype(float)

        y_true = y_true - np.mean(y_true)
        y_pred = y_pred - np.mean(y_pred)

        dot = float(np.sum(y_true * y_pred))
        norm_true = float(np.linalg.norm(y_true))
        norm_pred = float(np.linalg.norm(y_pred))

        return float(dot / (norm_true * norm_pred + eps_val))

    y_true = y_true - np.mean(y_true, axis=-1, keepdims=True)
    y_pred = y_pred - np.mean(y_pred, axis=-1, keepdims=True)

    dot = np.sum(y_true * y_pred, axis=-1)
    norm_true = np.linalg.norm(y_true, axis=-1)
    norm_pred = np.linalg.norm(y_pred, axis=-1)

    cos = dot / (norm_true * norm_pred + eps_val)

    return float(np.mean(cos))


def _compute_excess_mass_ecr(realVec, estiVec, threshold, eps=1e-8):
    """
    Excess-Mass Extreme Capture Rate.

    给定极端阈值 threshold：
        true_excess = max(y - threshold, 0)
        pred_excess = max(y_hat - threshold, 0)
        captured_excess = min(pred_excess, true_excess)

    ECR = sum(captured_excess) / sum(true_excess)

    含义：
        模型捕获了真实极端超额幅度的多少比例。
    """
    true_excess = np.maximum(realVec - threshold, 0.0)
    pred_excess = np.maximum(estiVec - threshold, 0.0)

    captured_excess = np.minimum(pred_excess, true_excess)

    total_true_excess = float(np.sum(true_excess))
    total_captured_excess = float(np.sum(captured_excess))

    ecr_count = int(np.sum(realVec >= threshold))

    if total_true_excess <= 0:
        ecr = np.nan
    else:
        ecr = float(total_captured_excess / (total_true_excess + eps))

    return ecr, ecr_count, total_true_excess, total_captured_excess


def compute_regression_metrics(realVec, estiVec, config, mode, valid_mask=None):
    """计算回归任务的误差指标。"""
    eps = 1e-8

    realVec = np.asarray(realVec, dtype=float)
    estiVec = np.asarray(estiVec, dtype=float)

    realVec, estiVec, _, valid_count = _apply_metric_valid_mask(
        realVec,
        estiVec,
        valid_mask=valid_mask,
        config=config,
    )

    absError = np.abs(estiVec - realVec)

    MAE = np.mean(absError)
    MSE = np.mean((realVec - estiVec) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(absError / (np.abs(realVec) + eps))
    COS = _mean_cosine_similarity(realVec, estiVec, eps)

    NMAE = np.sum(absError) / (np.sum(np.abs(realVec)) + eps)
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / (
        np.sqrt(np.sum(realVec ** 2)) + eps
    )

    # 计算不同相对误差阈值下的准确率
    thresholds = [0.01, 0.05, 0.10]
    Acc = [
        np.mean((absError < (np.abs(realVec) * threshold)).astype(float))
        for threshold in thresholds
    ]

    # ============================================================
    # Extreme Capture Rate, Excess-Mass Version
    # ============================================================
    finite_real = realVec[np.isfinite(realVec)]

    fallback_raw_q90 = (
        float(np.quantile(finite_real, 0.90))
        if finite_real.size > 0
        else 0.0
    )
    fallback_raw_q99 = (
        float(np.quantile(finite_real, 0.99))
        if finite_real.size > 0
        else fallback_raw_q90
    )

    raw_q90 = float(getattr(config, "raw_q90", fallback_raw_q90))
    raw_q99 = float(getattr(config, "raw_q99", fallback_raw_q99))

    ECR_q90, ecr_count_q90, ecr_true_excess_q90, ecr_captured_excess_q90 = (
        _compute_excess_mass_ecr(
            realVec=realVec,
            estiVec=estiVec,
            threshold=raw_q90,
            eps=eps,
        )
    )

    ECR_q99, ecr_count_q99, ecr_true_excess_q99, ecr_captured_excess_q99 = (
        _compute_excess_mass_ecr(
            realVec=realVec,
            estiVec=estiVec,
            threshold=raw_q99,
            eps=eps,
        )
    )

    # ============================================================
    # Tail metrics
    # ============================================================
    diff_tail_models = [
        "extreme_lstm_memo",
        "extreme_lstm",
        "mcann",
    ]

    use_diff_tail = str(getattr(config, "model", "")) in diff_tail_models

    if use_diff_tail:
        if realVec.shape[-1] > 1:
            fallback_q90 = np.quantile(
                np.abs(np.diff(realVec, axis=-1).reshape(-1)),
                0.90,
            )
        else:
            fallback_q90 = 0.0

        tail_q90 = float(getattr(config, "tail_q90", fallback_q90))

        if realVec.shape[-1] > 1:
            real_diff = np.diff(realVec, axis=-1)
            tail_mask = np.abs(real_diff) >= tail_q90
            tail_count = int(np.sum(tail_mask))

            aligned_abs_err = np.abs(estiVec[..., 1:] - realVec[..., 1:])
            aligned_sq_err = (estiVec[..., 1:] - realVec[..., 1:]) ** 2
            aligned_real_abs = np.abs(realVec[..., 1:])
        else:
            tail_mask = np.zeros_like(realVec, dtype=bool)
            tail_count = 0

            aligned_abs_err = absError
            aligned_sq_err = (estiVec - realVec) ** 2
            aligned_real_abs = np.abs(realVec)
    else:
        tail_q90 = float(getattr(config, "raw_q90", fallback_raw_q90))

        if realVec.ndim == 3 and realVec.shape[-1] == 1:
            tail_mask = realVec[..., 0] >= tail_q90
        else:
            tail_mask = realVec >= tail_q90

        tail_count = int(np.sum(tail_mask))

        aligned_abs_err = absError
        aligned_sq_err = (estiVec - realVec) ** 2
        aligned_real_abs = np.abs(realVec)

    if tail_count > 0:
        tail_abs_err = aligned_abs_err[tail_mask]
        tail_sq_err = aligned_sq_err[tail_mask]
        tail_real_abs = aligned_real_abs[tail_mask]

        Tail_MAE = float(np.mean(tail_abs_err))
        Tail_RMSE = float(np.sqrt(np.mean(tail_sq_err)))
        Tail_MAPE = float(np.mean(tail_abs_err / (tail_real_abs + eps)))
    else:
        Tail_MAE = np.nan
        Tail_RMSE = np.nan
        Tail_MAPE = np.nan

    all_metrics = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "MAPE": MAPE,
        "NMAE": NMAE,
        "NRMSE": NRMSE,
        "COS": COS,
        "ECR_q90": ECR_q90,
        "ECR_q99": ECR_q99,
        "ECR_count_q90": ecr_count_q90,
        "ECR_count_q99": ecr_count_q99,

        # 额外记录，方便检查 ECR 是怎么来的
        "ECR_true_excess_q90": ecr_true_excess_q90,
        "ECR_captured_excess_q90": ecr_captured_excess_q90,
        "ECR_true_excess_q99": ecr_true_excess_q99,
        "ECR_captured_excess_q99": ecr_captured_excess_q99,

        # Tail metrics
        "Tail_MAE": Tail_MAE,
        "Tail_RMSE": Tail_RMSE,
        "Tail_MAPE": Tail_MAPE,
        "Tail_q90": tail_q90,
        "Tail_count": tail_count,

        # 原始高值阈值
        "Raw_q90": raw_q90,
        "Raw_q99": raw_q99,

        # 兼容原有整体指标
        "Acc_10": Acc[2],
        "Valid_count": valid_count,
    }

    return all_metrics
