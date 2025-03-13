# coding : utf-8
# Author : yuxiang Zeng
# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from modules.load_data.create_window_dataset import create_window_dataset
from utils.data_scaler import get_scaler


def get_ts(dataset, config):
    df = pd.read_csv(f'./datasets/{dataset}/{dataset}.csv').to_numpy()
    # x, y = df[:, 1:], df[:, -1]

    if config.ts_var == 1:
        x, y = df[:, 1:], df[:, -1]
    else:
        x, y = df[:, -1].reshape(-1, 1), df[:, -1]

    print(x.shape, y.shape)

    # 根据训练集对input进行特征归一化
    scaler = get_scaler(y, config)
    y = scaler.transform(y)
    x = scaler.transform(x)

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    return X_window, y_window, scaler