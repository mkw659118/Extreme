# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text

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


def get_data(code_idx):
    address = os.listdir('./datasets/financial/')
    with open(f'./datasets/financial/{address[code_idx]}', 'rb') as f:
        data = pickle.load(f)
    return data

def generate_data(start_date, end_date, code_idx):
    # 数据库配置
    with open('./datasets/sql_token.pkl', 'rb') as f:
        DB_URI = pickle.load(f)
    engine = create_engine(DB_URI)
    DATA_JSON_PATH = './datasets/data.json'

    # 加载分类数据
    with open(DATA_JSON_PATH) as f:
        group_map = json.load(f)

    # 遍历所有股票类别
    for group_idx, (group_name, group_list) in enumerate(group_map.items()):
        for group_data in group_list:
            code_list = group_data['code_list']
            try:
                sql = text("""SELECT fund_code, date, nav FROM b_fund_nav_details_new WHERE fund_code IN :codes AND date BETWEEN :start AND :end ORDER BY date""")
                df = pd.read_sql_query(
                    sql.bindparams(codes=tuple(code_list), start=start_date, end=end_date),
                    engine
                )
            except Exception as e:
                print(f"数据库查询失败: {str(e)}")
                raise e

            # 为每组的股票做映射，获得分组可以去其他地方做
            df = df.to_numpy()
            unique_values = np.unique(df[:, 0])
            index_mapping = {value: idx for idx, value in enumerate(unique_values)}
            values = [[] for _ in range(len(unique_values))]
            dates = [[] for _ in range(len(unique_values))]
            for i in range(len(df)):
                idx = index_mapping[df[i][0]]
                dates[idx].append([df[i][1].year, df[i][1].month, df[i][1].day, df[i][1].weekday()])
                values[idx].append(float(df[i][2]))

            os.makedirs(f'./datasets/financial/', exist_ok=True)
            for key, value in index_mapping.items():
                now_data = np.concatenate([np.array([key] * len(dates[value])).reshape(-1, 1), np.array(dates[value]), np.array(values[value]).reshape(-1, 1)], axis=1)
                with open(f'./datasets/financial/{key}.pkl', 'wb') as f:
                    pickle.dump(now_data, f)
                    print(f'./datasets/financial/{key}.pkl 存储完毕')
    data = get_data(code_idx)
    return data


def get_financial_data(start_date, end_date, config):
    try:
        data = get_data(0)
        # print(data)
    except Exception as e:
        data = generate_data(start_date, end_date, 0)

    # print(data)
    x, y = data, data[:, -1].astype(np.float32)
    scaler = y[:int(len(x) * config.density)].astype(np.float32)
    y = (y - np.mean(scaler)) / np.std(scaler)
    # print(scaler.shape)
    x[:, -1] = x[:, -1].astype(np.float32)
    temp = x[:, -1].astype(np.float32)
    x[:, -1] = (temp - np.mean(scaler)) / np.std(scaler)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    return X_window, y_window

