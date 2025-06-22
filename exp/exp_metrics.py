# coding : utf-8
# Author : yuxiang Zeng

import torch as t
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def ErrorMetrics(realVec, estiVec, config):
    """ 根据任务类型选择合适的误差计算方式 """
    if isinstance(realVec, np.ndarray):
        realVec = realVec.astype(float)
    elif isinstance(realVec, t.Tensor):
        realVec = realVec.cpu().detach().numpy().astype(float)

    if isinstance(estiVec, np.ndarray):
        estiVec = estiVec.astype(float)
    elif isinstance(estiVec, t.Tensor):
        estiVec = estiVec.cpu().detach().numpy().astype(float)

    if config.classification:
        return compute_classification_metrics(realVec, estiVec)
    else:
        return compute_regression_metrics(realVec, estiVec)



def compute_regression_metrics(realVec, estiVec):
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

    return {
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'NMAE': NMAE,
        'NRMSE': NRMSE,
        'Acc_10': Acc[2],
    }


def compute_classification_metrics(realVec, estiVec):
    """ 计算分类任务的指标 """
    AC = accuracy_score(realVec, estiVec)
    PR = precision_score(realVec, estiVec, average='macro', zero_division=0)
    RC = recall_score(realVec, estiVec, average='macro', zero_division=0)
    F1 = f1_score(realVec, estiVec, average='macro')

    return {
        'AC': AC,
        'PR': PR,
        'RC': RC,
        'F1': F1,
    }
