import torch as t
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def ErrorMetrics(realVec, estiVec, config, mode):
    """ 根据任务类型选择合适的误差计算方式 """
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)

    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)
    
    return compute_regression_metrics(realVec, estiVec, config, mode)
        # return compute_regression_metrics_rolling(realVec, estiVec, config, config.pred_len)

def compute_regression_metrics_rolling(realVec, estiVec, config, rm):
    """ 计算回归任务的误差指标，支持滚动窗口评估 """
    # 转换为numpy数组
    realVec = np.array(realVec)
    estiVec = np.array(estiVec)
    
    # 先进行滚动窗口处理（使用完整数据）
    ll = int(len(estiVec) / config.pred_len)

    # print("estiVec length:", len(estiVec))
    # print("config.pred_len:", config.pred_len)

    # print("ll的长度")
    # print(ll)
    esti_all = []
    real_all = []
    for i in range(ll):
        # 按滚动窗口截取数据
        esti_window = estiVec[i * config.pred_len : (i * config.pred_len + rm)]
        real_window = realVec[i * config.pred_len : (i * config.pred_len + rm)]
        esti_all.extend(esti_window)
        real_all.extend(real_window)
    
    # 转换为numpy数组
    esti_all = np.array(esti_all)
    real_all = np.array(real_all)
    
    # 计算误差指标
    absError = np.abs(esti_all - real_all)

    MAE = np.mean(np.abs(real_all - esti_all))
    MSE = np.mean((real_all - esti_all) **2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((real_all - esti_all) / real_all))
    
    NMAE = np.sum(absError) / np.sum(np.abs(real_all))
    NRMSE = np.sqrt(np.sum((real_all - esti_all)** 2)) / np.sqrt(np.sum(real_all **2))

    # 计算不同阈值下的准确率
    thresholds = [0.01, 0.05, 0.10]
    Acc = [np.mean((absError < (real_all * t)).astype(float)) for t in thresholds]
    
    all_metrics = {
        'MAE_8': MAE,
        'MSE_8': MSE,
        'RMSE_8': RMSE,
        'MAPE_8': MAPE,
        'NMAE_8': NMAE,
        'NRMSE_8': NRMSE,
        'Acc_10%_8': Acc[2],
    }
    
    return all_metrics


def compute_regression_metrics(realVec, estiVec, config, mode):
    """ 计算回归任务的误差指标 """
    eps = 1e-8
    absError = np.abs(estiVec - realVec)

    # def mean_cosine_similarity(y_true, y_pred, eps_val=1e-8):
    #     y_true = np.asarray(y_true)
    #     y_pred = np.asarray(y_pred)
    #     if y_true.shape != y_pred.shape:
    #         raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    #     # Do not flatten: compute cosine along the last (time) dimension,
    #     # then average across samples (and channels if present).
    #     dot = np.sum(y_true * y_pred, axis=-1)
    #     norm_true = np.linalg.norm(y_true, axis=-1)
    #     norm_pred = np.linalg.norm(y_pred, axis=-1)
    #     cos = dot / (norm_true * norm_pred + eps_val)
    #     return float(np.mean(cos))
    def mean_cosine_similarity(y_true, y_pred, eps_val=1e-8):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

        if y_true.ndim == 3 and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
            y_pred = y_pred[..., 0]

        # Center each sample before computing cosine so that shared level/baseline
        # does not dominate the similarity score.
        y_true = y_true - np.mean(y_true, axis=-1, keepdims=True)
        y_pred = y_pred - np.mean(y_pred, axis=-1, keepdims=True)

        dot = np.sum(y_true * y_pred, axis=-1)
        norm_true = np.linalg.norm(y_true, axis=-1)
        norm_pred = np.linalg.norm(y_pred, axis=-1)
        cos = dot / (norm_true * norm_pred + eps_val)
        return float(np.mean(cos))

    MAE = np.mean(np.abs(realVec - estiVec))
    MSE = np.mean((realVec - estiVec) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((realVec - estiVec) / (np.abs(realVec) + eps)))
    COS = mean_cosine_similarity(realVec, estiVec, eps)
    
    NMAE = np.sum(absError) / np.sum(np.abs(realVec))
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))

    # 计算不同阈值下的准确率
    thresholds = [0.01, 0.05, 0.10]
    Acc = [np.mean((absError < (realVec * t)).astype(float)) for t in thresholds]

    # Tail metrics (q90): threshold/mask are both based on |diff|.
    fallback_q90 = np.quantile(np.abs(np.diff(realVec, axis=-1).reshape(-1)), 0.90) if realVec.shape[-1] > 1 else 0.0
    tail_q90 = float(getattr(config, 'tail_q90', fallback_q90))
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
        aligned_abs_err = np.abs(estiVec - realVec)
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

    # Extreme Capture Rate (ECR) on raw values:
    # point is extreme if y >= q; captured if |y_hat - y| <= epsilon(y),
    # where epsilon(y)=max(alpha*(q99-q90), rho*|y|).
    finite_real = realVec[np.isfinite(realVec)]
    fallback_raw_q90 = float(np.quantile(finite_real, 0.90)) if finite_real.size > 0 else 0.0
    fallback_raw_q99 = float(np.quantile(finite_real, 0.99)) if finite_real.size > 0 else fallback_raw_q90
    raw_q90 = float(getattr(config, 'raw_q90', fallback_raw_q90))
    raw_q99 = float(getattr(config, 'raw_q99', fallback_raw_q99))

    ecr_alpha = float(getattr(config, 'ecr_alpha', 0.2))
    ecr_rho = float(getattr(config, 'ecr_rho', 0.02))
    eps_base = max(0.0, ecr_alpha * (raw_q99 - raw_q90))
    eps_dyn = np.maximum(eps_base, ecr_rho * np.abs(realVec))

    extreme_mask_q90 = realVec >= raw_q90
    extreme_mask_q99 = realVec >= raw_q99
    captured_mask = absError <= eps_dyn

    ecr_count_q90 = int(np.sum(extreme_mask_q90))
    ecr_count_q99 = int(np.sum(extreme_mask_q99))
    ECR_q90 = float(np.mean(captured_mask[extreme_mask_q90])) if ecr_count_q90 > 0 else np.nan
    ECR_q99 = float(np.mean(captured_mask[extreme_mask_q99])) if ecr_count_q99 > 0 else np.nan

    all_metrics_72 = {
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'COS': COS,
        'Tail_MAE': Tail_MAE,
        'Tail_RMSE': Tail_RMSE,
        'Tail_MAPE': Tail_MAPE,
        'Tail_q90': tail_q90,
        'Tail_count': tail_count,
        'Raw_q90': raw_q90,
        'Raw_q99': raw_q99,
        'ECR_eps_base': float(eps_base),
        'ECR_alpha': ecr_alpha,
        'ECR_rho': ecr_rho,
        'ECR_q90': ECR_q90,
        'ECR_q99': ECR_q99,
        'ECR_count_q90': ecr_count_q90,
        'ECR_count_q99': ecr_count_q99,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'Acc_10': Acc[2],
    }

    

    # if mode == 'valid':
    #     return all_metrics_72
    # else:
    # all_metrics_8 = compute_regression_metrics_rolling(realVec, estiVec, config, rm=8)
    # all_metrics = {**all_metrics_72, **all_metrics_8}
    return all_metrics_72

    # return all_metrics_72
