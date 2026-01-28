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
    absError = np.abs(estiVec - realVec)

    MAE = np.mean(np.abs(realVec - estiVec))
    MSE = np.mean((realVec - estiVec) ** 2)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((realVec - estiVec) / realVec))
    
    NMAE = np.sum(absError) / np.sum(np.abs(realVec))
    NRMSE = np.sqrt(np.sum((realVec - estiVec) ** 2)) / np.sqrt(np.sum(realVec ** 2))

    # 计算不同阈值下的准确率
    thresholds = [0.01, 0.05, 0.10]
    Acc = [np.mean((absError < (realVec * t)).astype(float)) for t in thresholds]
    all_metrics_72 = {
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAPE': MAPE,
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
