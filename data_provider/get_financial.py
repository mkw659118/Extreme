# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from data_provider.generate_financial import get_all_fund_list, generate_data, process_fund
from utils.data_scaler import get_scaler

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
    with open('./results/func_code_to_label_30.pkl', 'rb') as f:
        data = pickle.load(f)
    all_func_code = []
    for i in range(len(data)):
        if data[i][1] == group_index:
            all_func_code.append(data[i][0])
    return all_func_code

def multi_dataset(config):
    # now_fund_code = get_benchmark_code()
    now_fund_code = get_group_idx(27)
    min_length = 1e9
    all_data = []
    for fund_code in tqdm(now_fund_code, desc='SQL'):
        try:
            # df = get_data(config.start_date, config.end_date, fund_code)
            df = process_fund(0, fund_code, config.start_date, config.end_date)
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
    scaler = get_scaler(y, config)
    return x, y, scaler


def get_financial_data(start_date, end_date, idx, config):
    # now_fund_code = get_all_fund_list()[idx]
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


def filter_jump_sequences(X_window, y_window, threshold=0.3, mode='absolute'):
    """
    X_window: [n, seq_len, d] numpy array，其中最后一个维度是 value
    y_window: [n, ...]，标签
    返回：
        - 过滤后的 X_window
        - 对应的 y_window
        - 保留的索引 idx（相对于原始）
    """
    values = X_window[:, :, -1]  # 提取 value 部分
    diff = values[:, 1:] - values[:, :-1]

    if mode == 'absolute':
        mask = np.any(np.abs(diff) > threshold, axis=1)
    elif mode == 'relative':
        prev = np.clip(values[:, :-1], 1e-5, None)
        rel_diff = np.abs(diff / prev)
        mask = np.any(rel_diff > threshold, axis=1)
    else:
        raise ValueError("mode must be 'absolute' or 'relative'")
    idx = np.where(~mask)[0]
    X_window, y_window = X_window[idx], y_window[idx]
    return X_window, y_window





# def multi_dataset(config):
#     now_fund_code = get_benchmark_code()
#     all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y = [], [], [], [], [], []
#     for i in range(len(now_fund_code)):
#         config.idx = i
#         config.multi_dataset = False
#         datamodule = a_data_center.DataModule(config)
#         if len(datamodule.train_set.x) == 0 or len(datamodule.y) <= config.seq_len:
#             continue
#         all_train_x.append(datamodule.train_set.x)
#         all_train_y.append(datamodule.train_set.y)
#
#         all_valid_x.append(datamodule.valid_set.x)
#         all_valid_y.append(datamodule.valid_set.y)
#
#         all_test_x.append(datamodule.test_set.x)
#         all_test_y.append(datamodule.test_set.y)
#         del datamodule
#
#     all_train_x = np.concatenate(all_train_x, axis=0)
#     all_train_y = np.concatenate(all_train_y, axis=0)
#
#     all_valid_x = np.concatenate(all_valid_x, axis=0)
#     all_valid_y = np.concatenate(all_valid_y, axis=0)
#
#     all_test_x = np.concatenate(all_test_x, axis=0)
#     all_test_y = np.concatenate(all_test_y, axis=0)
#
#     y_mean = np.mean(all_train_y)
#     y_std = np.std(all_train_y)
#
#     all_train_y = (all_train_y - y_mean) / y_std
#     all_valid_y = (all_valid_y - y_mean) / y_std
#     all_test_y = (all_test_y - y_mean) / y_std
#
#     for i in range(len(input_keys)):
#         all_train_x[:, -i] = all_train_x[:, -i].astype(np.float32)
#         all_valid_x[:, -i] = all_valid_x[:, -i].astype(np.float32)
#         all_test_x[:, -i] = all_test_x[:, -i].astype(np.float32)
#
#         now_mean = np.mean(all_train_x[:, -i])
#         now_std = np.std(all_train_x[:, -i]) + 1e-9
#
#         all_train_x[:, -i] = (all_train_x[:, -i] - now_mean) / now_std
#         all_valid_x[:, -i] = (all_valid_x[:, -i] - now_mean) / now_std
#         all_test_x[:, -i] = (all_test_x[:, -i] - now_mean) / now_std
#
#     scaler = get_scaler(all_train_y, config)
#     scaler.y_mean, scaler.y_std = y_mean, y_std
#     print(all_train_x.shape, all_train_y.shape)
#
#     return all_train_x, all_train_y, all_valid_x, all_valid_y, all_test_x, all_test_y, scaler
