# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年04月22日11:54:01）
import numpy as np


def preprocess_data(x, y, config):
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    return x, y


def parse_split_ratio(ratio_str):
    """解析如 '7:1:2' 的字符串为归一化比例 [0.7, 0.1, 0.2]"""
    parts = list(map(int, ratio_str.strip().split(':')))
    total = sum(parts)
    return [p / total for p in parts]


def get_split_dataset(x, y, config):
    if config.classification:
        return get_train_valid_test_classification_dataset(x, y, config)
    else:
        return get_train_valid_test_dataset(x, y, config)


def get_train_valid_test_dataset(x, y, config):
    x, y = preprocess_data(x, y, config)
    train_ratio, valid_ratio, test_ratio = parse_split_ratio(config.spliter_ratio)
    if config.shuffle:
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]
    if config.use_train_size:
        train_size = int(config.train_size)
    else:
        train_size = int(len(x) * train_ratio)

    if config.eval_set:
        valid_size = int(len(x) * valid_ratio)
    else:
        valid_size = 0

    train_x = x[:train_size]
    train_y = y[:train_size]
    valid_x = x[train_size:train_size + valid_size]
    valid_y = y[train_size:train_size + valid_size]
    test_x = x[train_size + valid_size:]
    test_y = y[train_size + valid_size:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_train_valid_test_classification_dataset(x, y, config):
    x, y = preprocess_data(x, y, config)
    train_ratio, valid_ratio, test_ratio = parse_split_ratio(config.spliter_ratio)
    from collections import defaultdict
    import random
    class_data = defaultdict(list)
    for now_x, now_label in zip(x, y):
        class_data[now_label].append(now_x)
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []
    for label, now_x in class_data.items():
        random.shuffle(now_x)
        if config.use_train_size:
            train_size = int(config.train_size)
        else:
            train_size = int(len(x) * train_ratio)
        valid_size = int(len(now_x) * valid_ratio) if config.eval_set else 0
        train_x.extend(now_x[:train_size])
        train_y.extend([label] * len(now_x[:train_size]))
        valid_x.extend(now_x[train_size:train_size + valid_size])
        valid_y.extend([label] * len(now_x[train_size:train_size + valid_size]))
        test_x.extend(now_x[train_size + valid_size:])
        test_y.extend([label] * len(now_x[train_size + valid_size:]))
    return train_x, train_y, valid_x, valid_y, test_x, test_y
