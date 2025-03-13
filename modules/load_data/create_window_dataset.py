import numpy as np


def create_window_dataset(data, target, window_size, forecast_size):
    """
    将时间序列数据转换为窗口数据集
    :param data: 特征数据（numpy数组）
    :param target: 目标数据（numpy数组）
    :param window_size: 输入窗口大小（历史时间步数）
    :param forecast_size: 预测窗口大小（未来时间步数）
    :return: 窗口化的特征数据和目标数据
    """
    X, y = [], []
    total_samples = len(data) - window_size - forecast_size + 1

    for i in range(total_samples):
        # 获取当前窗口的特征数据
        X_window = data[i:i + window_size]
        # 获取对应的未来窗口目标数据
        y_window = target[i + window_size: i + window_size + forecast_size]

        X.append(X_window)
        y.append(y_window)

    return np.array(X), np.array(y)
