# coding : utf-8
# Author : yuxiang Zeng
# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def norm(x):
    return np.mean(x), np.std(x)

def get_ts(dataset, config):
    df = pd.read_csv(f'./datasets/{dataset}/{dataset}.csv').to_numpy()
    # x, y = df[:, 1:], df[:, -1]

    if config.ts_var == 1:
        x, y = df[:, 1:], df[:, -1]
    else:
        x, y = df[:, -1].reshape(-1, 1), df[:, -1]

    print(x.shape, y.shape)

    # 根据训练集对input进行特征归一化
    scaler = y[:int(len(x) * config.density)]
    # scaler = StandardScaler()
    # scaler.fit(train_x)
    # x = scaler.transform(x)
    # y = scaler.transform(y)
    x = (x - np.mean(scaler)) / np.std(scaler)
    y = (y - np.mean(scaler)) / np.std(scaler)

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    return X_window, y_window