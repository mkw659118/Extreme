# coding : utf-8
# Author : yuxiang Zeng
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from modules.load_data.get_ts import get_ts
from utils.logger import Logger
from utils.plotter import MetricsPlotter
from utils.utils import set_settings
from tqdm import *
import pickle
from utils.config import get_config



class experiment:
    def __init__(self, config):
        self.config = config

    def load_data(self, config):
        all_x, all_y = get_ts(config.dataset, config)
        return all_x, all_y


    def get_pytorch_index(self, data):
        return torch.as_tensor(data)


# 数据集定义
class DataModule:
    def __init__(self, exper_type, config):
        self.config = config
        self.path = config.path
        self.x, self.y = exper_type.load_data(config)
        if config.debug:
            self.x, self.y = self.x[:300], self.y[:300]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.max_value = (
            self.get_train_valid_test_dataset(self.x, self.y, config)) if not config.classification else self.get_train_valid_test_classification_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {self.max_value:.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )


    def preprocess_data(self, x, y, config):
        x = np.array(x)
        y = np.array(y)
        return x, y

    def get_train_valid_test_dataset(self, x, y, config):
        x, y = self.preprocess_data(x, y, config)
        # indices = np.random.permutation(len(x))
        # x, y = x[indices], y[indices]
        if not config.classification:
            # 2025年2月23日00:48:26
            # max_value = y.max()
            # y /= max_value
            max_value = 1
        else:
            max_value = 1
        train_size = int(len(x) * config.density)
        if config.eval_set:
            valid_size = int(len(x) * 0.10)
        else:
            valid_size = 0

        train_y = y[:train_size]
        df_mean = np.mean(train_y)
        df_std = np.std(train_y)
        x = (x - df_mean) / df_std
        y = (y - df_mean) / df_std

        train_x = x[:train_size]
        train_y = y[:train_size]

        valid_x = x[train_size:train_size + valid_size]
        valid_y = y[train_size:train_size + valid_size]
        test_x = x[train_size + valid_size:]
        test_y = y[train_size + valid_size:]
        return train_x, train_y, valid_x, valid_y, test_x, test_y, max_value


    def get_train_valid_test_classification_dataset(self, x, y, config):
        x, y = self.preprocess_data(x, y, config)
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
        max_value = 1
        return train_x, train_y, valid_x, valid_y, test_x, test_y, max_value


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y


def custom_collate_fn(batch, config):
    from torch.utils.data.dataloader import default_collate
    x, y = zip(*batch)
    x, y = default_collate(x), default_collate(y)
    return x, y


def get_dataloaders(train_set, valid_set, test_set, config):
    import platform
    from torch.utils.data import DataLoader
    import multiprocessing
    if platform.system() == 'Linux' and 'ubuntu' in platform.version().lower():
        max_workers = multiprocessing.cpu_count() // 5
        prefetch_factor = 4
    else:
        max_workers = 0
        prefetch_factor = None

    train_loader = DataLoader(
        train_set,
        batch_size=config.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, config),
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, config),
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, config),
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':

    config = get_config()
    set_settings(config)
    config.experiment = True

    # logger plotter
    exper_detail = f"Dataset : {config.dataset.upper()}, Model : {config.model}, Train_size : {config.train_size}"
    log_filename = f'{config.train_size}_r{config.rank}'
    log = Logger(log_filename, exper_detail, config)
    plotter = MetricsPlotter(log_filename, config)
    config.log = log
    log(str(config.__dict__))


    exper = experiment(config)
    datamodule = DataModule(exper, config)
    for train_batch in datamodule.train_loader:
        all_item = [item.to(config.device) for item in train_batch]
        inputs, label = all_item[:-1], all_item[-1]
        print(num_windowss.shape, value.shape)
        # break
    print('Done!')
