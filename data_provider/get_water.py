#Author  :   mkw 
#Time    :   2025/09/08 21:07:02
#Desc    :   None

import numpy as np
import pandas as pd
from data_provider.data_scaler import get_scaler


def get_water(dataset, config):
    df = pd.read_csv(f'./datasets/' + config.reservoir_sensor + '.tsv', sep='\t').to_numpy()
    # x, y = df[:, 1:], df[:, -1]
    if config.water_var == 1:  #多变量预测
        x, y = df[:, 1:], df[:, 1:]
    else:                   #单变量预测
        x, y = df[:, -1], df[:, -1]

    # 把第一列转成 datetime
    timestamps = pd.to_datetime(df[:, 0])
    timestamps = np.array([[ts.month, ts.day, ts.weekday(), ts.hour] for ts in timestamps])

    # 根据训练集对input进行特征归一化
    y_scaler = get_scaler(y, config)
    y = y_scaler.transform(y)

    x_scaler = get_scaler(x, config)
    x = x_scaler.transform(x)

    x = np.concatenate((timestamps, x), axis=1)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    return x, y, x_scaler, y_scaler