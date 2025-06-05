# coding : utf-8
# Author : yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年04月22日11:54:01）
import numpy as np

def preprocess_data(x, y, config):
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    return x, y

def get_split_dataset(x, y, config):
    if config.classification:
        return get_train_valid_test_classification_dataset(x, y, config)
    else:
        return get_train_valid_test_dataset(x, y, config)

def get_train_valid_test_dataset(x, y, config):
    x, y = preprocess_data(x, y, config)
    if config.shuffle:
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

    x_len = len(x) - config.seq_len - config.pred_len + 1
    # x_len = len(x)
    if config.use_train_size:
        train_size = int(config.train_size)
    else:
        train_size = int(x_len * config.density)

    if config.eval_set:
        valid_size = int((int(x_len) - train_size) / 2)
    else:
        valid_size = 0
    print(x_len, train_size, valid_size, x_len - train_size - valid_size)

    train_x = x[:train_size]
    train_y = y[:train_size]
    valid_x = x[train_size:train_size + valid_size]
    valid_y = y[train_size:train_size + valid_size]
    test_x = x[train_size + valid_size:]
    test_y = y[train_size + valid_size:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_train_valid_test_classification_dataset(x, y, config):
    x, y = preprocess_data(x, y, config)
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
            train_size = int(len(x) * config.density)
        valid_size = int(len(now_x) * 0.10) if config.eval_set else 0
        train_x.extend(now_x[:train_size])
        train_y.extend([label] * len(now_x[:train_size]))
        valid_x.extend(now_x[train_size:train_size + valid_size])
        valid_y.extend([label] * len(now_x[train_size:train_size + valid_size]))
        test_x.extend(now_x[train_size + valid_size:])
        test_y.extend([label] * len(now_x[train_size + valid_size:]))
    return train_x, train_y, valid_x, valid_y, test_x, test_y