# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text

import data_center
from modules.load_data.create_window_dataset import create_window_dataset


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


def get_financial_data(start_date, end_date, idx, config):
    try:
        data = get_data(idx)
    except Exception as e:
        data = generate_data(start_date, end_date, idx)
    # 过滤掉数据库存储数据异常 2025年3月17日10:30:47
    data = np.stack([df for df in data if int(df[1]) <= int(end_date.split('-')[0])])
    data = data.astype(np.float32)

    for i in range(len(data)):
        if int(data[i][1]) > 2025:
            print(data[i])

    x, y = data, data[:, -1].astype(np.float32)
    scaler = None
    if not config.multi_dataset:
        scaler = get_scaler(y, config)
        y = scaler.transform(y)
        x[:, -1] = x[:, -1].astype(np.float32)
        temp = x[:, -1].astype(np.float32)
        x[:, -1] = (temp - scaler.y_mean) / scaler.y_std

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    for i in range(len(X_window)):
        if X_window[i][0][1] > 2025:
            print(X_window[i][0])
    return X_window, y_window, scaler


from utils.data_scaler import get_scaler

def normorlize(data, value_mean, value_std):
    # print(data.shape)
    data[:, :, -1] = data[:, :, -1].astype(np.float32)
    temp = data[:, :, -1].astype(np.float32)
    data[:, :, -1] = (temp - value_mean) / value_std
    return data

def check_data():
    return

def multi_dataset(config):
    all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y = [], [], [], [], [], []
    for i in range(50):
        config.idx = i
        config.multi_dataset = False
        datamodule = data_center.DataModule(config)
        if len(datamodule.train_set.x) == 0 or len(datamodule.y) <= config.seq_len:
            continue
        all_train_x.append(datamodule.train_set.x)
        all_train_y.append(datamodule.train_set.y)

        all_valid_x.append(datamodule.valid_set.x)
        all_valid_y.append(datamodule.valid_set.y)

        all_test_x.append(datamodule.test_set.x)
        all_test_y.append(datamodule.test_set.y)
        del datamodule

    all_train_x = np.concatenate(all_train_x, axis=0)
    all_train_y = np.concatenate(all_train_y, axis=0)

    all_valid_x = np.concatenate(all_valid_x, axis=0)
    all_valid_y = np.concatenate(all_valid_y, axis=0)

    all_test_x = np.concatenate(all_test_x, axis=0)
    all_test_y = np.concatenate(all_test_y, axis=0)

    for i in range(len(all_train_x)):
        if all_train_x[i][0][1] > 2025:
            print(all_train_x[i][0])
            # exit()

    y_mean = np.mean(all_train_y)
    y_std = np.std(all_train_y)

    all_train_y = (all_train_y - y_mean) / y_std
    all_valid_y = (all_valid_y - y_mean) / y_std
    all_test_y = (all_test_y - y_mean) / y_std

    all_train_x = normorlize(all_train_x, y_mean, y_std)
    all_valid_x = normorlize(all_valid_x, y_mean, y_std)
    all_test_x = normorlize(all_test_x, y_mean, y_std)

    scaler = get_scaler(all_train_y, config)
    scaler.y_mean, scaler.y_std = y_mean, y_std
    print(all_train_x.shape, all_train_y.shape)
    return all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y, scaler


