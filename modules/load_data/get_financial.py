# coding : utf-8
# Author : yuxiang Zeng
import json
import os
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine, text

def save_data(df, groupid):
    df = df.to_numpy()
    unique_values = np.unique(df[:, 0])
    all_data = {}
    index_mapping = {value: idx for idx, value in enumerate(unique_values)}
    values = [[] for _ in range(len(unique_values))]
    dates = [[] for _ in range(len(unique_values))]
    for i in range(len(df)):
        idx = index_mapping[df[i][0]]
        dates[idx].append([df[i][1].year, df[i][1].month, df[i][1].day, df[i][1].weekday()])
        values[idx].append(df[i][2])

    for i in range(len(values)):
        print(len(dates[i]))

    os.makedirs(f'./datasets/financial/{groupid}', exist_ok=True)
    for key, value in index_mapping.items():
        # print(key, value)
        all_data[key] = dates[value], values[value]
        with open(f'./datasets/financial/{groupid}/{key}.pkl', 'wb') as f:
            pickle.dump(all_data, f)
            print(f'./datasets/financial/{groupid}/{key}.pkl 存储完毕')
    return True


def get_financial_data(group_idx, group_data, start_date, end_date):
    """处理单个group的时间序列数据"""
    # 获取代码列表
    code_list = group_data['code_list']
    if not code_list:
        return None, None

    # 数据库查询
    try:
        sql = text("""SELECT fund_code, date, nav FROM b_fund_nav_details_new WHERE fund_code IN :codes AND date BETWEEN :start AND :end ORDER BY date""")
        df = pd.read_sql_query(
            sql.bindparams(codes=tuple(code_list), start=start_date, end=end_date),
            engine
        )
    except Exception as e:
        print(f"数据库查询失败: {str(e)}")
        return None, None

    save_data(df, group_idx)

    # 时间轴处理
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_time = full_dates.strftime('%Y-%m-%d').values.reshape(-1, 1)

    # 创建三维矩阵 [n, time, 1]
    nav_seq = np.full((len(code_list), len(full_dates), 1), np.nan)

    # 填充数据
    for i, code in enumerate(code_list):
        code_df = df[df['fund_code'] == code]
        if not code_df.empty:
            # 合并完整时间序列
            merged = pd.DataFrame(index=full_dates)
            merged['nav'] = code_df.set_index('date')['nav']
            nav_seq[i, :, 0] = merged['nav'].ffill().values

    return nav_seq.astype(np.float32), date_time


def process_group(group_idx, group_data, start_date, end_date):
    """处理单个group并保存数据"""
    print(f"Processing group {group_idx}...")

    try:
        matrix = np.load(os.path.join(SAVE_DIR, f'group{group_idx}.npy'))
        dates = np.load(os.path.join(SAVE_DIR, f'group{group_idx}.npy'))
        print(f"Group {group_idx} 读取成功，形状: {matrix.shape}")
    except Exception as e:
        # 执行核心处理
        matrix, dates = get_financial_data(group_idx, group_data, start_date, end_date)
        if matrix is not None:
            # 保存数据
            os.makedirs(SAVE_DIR, exist_ok=True)
            # np.save(os.path.join(SAVE_DIR, f'group{group_idx}.npy'), matrix)
            # np.save(os.path.join(SAVE_DIR, f'datetime{group_idx}.npy'), dates)
            # print(f"Group {group_idx} 保存成功，形状: {matrix.shape}")
        else:
            print(f"Group {group_idx} 处理失败")

    return matrix, dates

def main(start_date, end_date):
    # 加载分类数据
    with open(DATA_JSON_PATH) as f:
        group_map = json.load(f)

    # 遍历所有股票类别
    for group_idx, (group_name, group_list) in enumerate(group_map.items()):
        for sub_group in group_list:
            matrix, dates = process_group(group_idx, sub_group, start_date, end_date)


if __name__ == "__main__":
    # 数据库配置
    DB_URI = 'mysql+pymysql://root:qilai123@123.57.74.222:3306/fund'
    engine = create_engine(DB_URI)

    # 路径配置
    DATA_JSON_PATH = './datasets/data.json'
    SAVE_DIR = './datasets/financial/'

    # 时间范围配置
    start = '2022-07-13'
    end = '2023-07-13'

    # 执行处理流程
    main(start, end)