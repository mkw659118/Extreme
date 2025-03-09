# coding : utf-8
# Author : yuxiang Zeng
from utils.data_preprocessor import preprocess_data
import numpy as np

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
    train_size = int(len(x) * config.density)
    if config.eval_set:
        valid_size = int(len(x) * 0.10)
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
        train_size = int(len(now_x) * config.density)
        valid_size = int(len(now_x) * 0.10) if config.eval_set else 0
        train_x.extend(now_x[:train_size])
        train_y.extend([label] * len(now_x[:train_size]))
        valid_x.extend(now_x[train_size:train_size + valid_size])
        valid_y.extend([label] * len(now_x[train_size:train_size + valid_size]))
        test_x.extend(now_x[train_size + valid_size:])
        test_y.extend([label] * len(now_x[train_size + valid_size:]))
    return train_x, train_y, valid_x, valid_y, test_x, test_y