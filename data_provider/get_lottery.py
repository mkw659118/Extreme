# coding : utf-8
# Author : Yuxiang Zeng

import pandas as pd
import numpy as np
from utils.data_scaler import get_scaler


def get_lottery(datasets, config):
    df = pd.read_excel('./datasets/lottery/pl3_desc.xls', header=1)
    df = df.iloc[::-1].reset_index(drop=True)
    timestamps = pd.to_datetime(df['开奖日期'])
    timestamps = np.array([[ts.year, ts.month, ts.day, ts.weekday()] for ts in timestamps])
    x = np.array(df['号']).reshape(-1, 1)
    # timestamps
    x = np.concatenate((timestamps, x), axis=-1)
    y = np.array(df['号']).reshape(-1, )

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
    print(X_window.shape, y_window.shape)
    return X_window, y_window, scaler
