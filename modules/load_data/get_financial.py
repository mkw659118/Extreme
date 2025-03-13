# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text

from modules.load_data.create_window_dataset import create_window_dataset
from utils.data_scaler import get_scaler


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
    scaler = get_scaler(y, config)
    y = scaler.transform(y)

    x[:, -1] = x[:, -1].astype(np.float32)
    temp = x[:, -1].astype(np.float32)
    x[:, -1] = (temp - scaler.y_mean) / scaler.y_std

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    return X_window, y_window, scaler

