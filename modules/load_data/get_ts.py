# coding : utf-8
# Author : yuxiang Zeng
# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from utils.data_scaler import get_scaler


def get_ts(dataset, config):
    df = pd.read_csv(f'./datasets/{dataset}/{dataset}.csv').to_numpy()
    # x, y = df[:, 1:], df[:, -1]
    if config.ts_var == 1:
        x, y = df[:, 1:], df[:, -1]
    else:
        x, y = df[:, -1].reshape(-1, 1), df[:, -1]

    # 把第一列转成 datetime
    timestamps = pd.to_datetime(df[:, 0])
    timestamps = np.array([[ts.year, ts.month, ts.day, ts.weekday()] for ts in timestamps])

    x = np.concatenate((timestamps, x), axis=1)
    print(x[0])
    print(x.shape, y.shape)

    # 根据训练集对input进行特征归一化
    scaler = get_scaler(y, config)
    y = scaler.transform(y)
    # x = scaler.transform(x)
    x[:, -1] = x[:, -1].astype(np.float32)
    temp = x[:, -1].astype(np.float32)
    x[:, -1] = (temp - scaler.y_mean) / scaler.y_std

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = x, y
    return X_window, y_window, scaler