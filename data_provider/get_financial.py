# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from data_provider.generate_financial import get_all_fund_list, generate_data, process_fund
from data_provider.data_scaler import get_scaler

input_keys = ['nav', 'accnav', 'adj_nav']
pred_value = 'nav'  # 'nav', 'accnav', 'adj_nav'

def get_data(start_date, end_date, fund_code):
    # 读取数据
    dir_name = 'S' + (start_date + '_E' + end_date).replace('-', '')

    with open(f'./datasets/financial/{dir_name}/{fund_code}.pkl', 'rb') as f:
        df = pickle.load(f)  # 假设 df 现在是 numpy 数组

    print(f'./datasets/financial/{dir_name}/{fund_code}.pkl')
    # 假设原始数据按顺序排列： fund_code, year, month, day, weekday, nav, accnav, adj_nav
    # 你已经知道 'nav', 'accnav', 'adj_nav' 列的位置：[-3, -2, -1]
    col_indices = {'nav': -3, 'accnav': -2, 'adj_nav': -1}

    # 提取 input_keys 中的列
    input_columns = [df[:, col_indices[key]] for key in input_keys if key != pred_value]

    # 获取其它列（fund_code, year, month, day, weekday）
    fund_code_column = df[:, 0]  # 第一列 fund_code
    year_column = df[:, 1]  # 第二列 year
    month_column = df[:, 2]  # 第三列 month
    day_column = df[:, 3]  # 第四列 day
    weekday_column = df[:, 4]  # 第五列 weekday

    # 将前五列和 input_keys 中的列拼接（不包括 pred_value）
    ordered_data = np.column_stack((fund_code_column, year_column, month_column, day_column, weekday_column, *input_columns))

    # 获取 pred_value 列并将其移到最后
    pred_column = df[:, col_indices[pred_value]]

    # 最终将目标列 pred_value 插入到数据末尾
    final_data = np.column_stack((ordered_data, pred_column))
    # print(final_data[:, -4:])
    return final_data

# 为了对齐实验，现在加上this one  20250513 15时47分
def get_benchmark_code():
    with open('./datasets/benchmark.pkl', 'rb') as f:
        group = pickle.load(f)
        group = group['stock-270000']
        # group.remove('013869')
        # group.remove('013870')
    # print(f'now fund code: {group}')
    return group


def get_group_idx(group_index):
    # './results/func_code_to_label_{n_clusters}.pkl'
    with open('./results/func_code_to_label_40_balanced.pkl', 'rb') as f:
        data = pickle.load(f)
    all_func_code = []
    for i in range(len(data)):
        if data[i][1] == group_index:
            all_func_code.append(data[i][0])
    return all_func_code


def multi_dataset(config):
    # now_fund_code = get_benchmark_code()
    now_fund_code = get_group_idx(config.idx)
    min_length = 1e9
    all_data = []
    for fund_code in tqdm(now_fund_code, desc='SQL'):
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            df = process_fund(0, fund_code, config.start_date, config.end_date)
            print(f"{fund_code} -- len = {len(df)}, now min_length = {min_length}")
            min_length = min(len(df), min_length)
            all_data.append(df)
        except Exception as e:
            print(e)
            process_fund(0, fund_code, config.start_date, config.end_date)

    raw_data = []
    for df in all_data:
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            # df = process_fund(0, fund_code, config.start_date, config.end_date)
            raw_data.append(df[-min_length:])
        except Exception as e:
            print(e)
    data = np.stack(raw_data, axis=0)
    data = data.transpose(1, 0, 2)
    x, y = data[:, :, :], data[:, :, -1]

    x[:, :, -3:] = x[:, :, -3:].astype(np.float32)
    x_scaler = get_scaler(x[:, :, -3:], config, 'None')
    x[:, :, -3:] = x_scaler.transform(x[:, :, -3:])

    y_scaler = get_scaler(y, config, 'None')
    y = y_scaler.transform(y)
    return x, y, x_scaler, y_scaler


def get_financial_data(start_date, end_date, idx, config):
    # fund_code = get_all_fund_list()[idx]
    # 为了对齐实验，现在加上this one  20250513 15时47分
    fund_code = get_benchmark_code()[idx]
    try:
        data = get_data(start_date, end_date, fund_code)
        # data = process_fund(0, fund_code, config.start_date, config.end_date)
    except Exception as e:
        print(e)
        generate_data(start_date, end_date)
        data = get_data(start_date, end_date, fund_code)

    data = data.astype(np.float32)

    x, y = data, data[:, -1].astype(np.float32)
    scaler = None

    if not config.multi_dataset:
        for i in range(len(input_keys)):
            x[:, -i] = x[:, -i].astype(np.float32)
            scaler = get_scaler(x[:, -i], config)
            x[:, -i] = scaler.transform(x[:, -i])

        scaler = get_scaler(y, config)
        y = scaler.transform(y)

    y = y.astype(np.float32)
    # 构建
    X_window, y_window = x, y
    # X_window, y_window = create_window_dataset(x, y, config.seq_len, config.pred_len)
    print(X_window.shape, y_window.shape)
    return X_window, y_window, scaler





