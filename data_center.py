# coding : utf-8
# Author : yuxiang Zeng
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_dataset import TensorDataset
from modules.load_data.get_financial import get_financial_data
from modules.load_data.get_ts import get_ts
from utils.data_dataloader import get_dataloaders
from utils.data_spliter import get_split_dataset
from utils.exp_logger import Logger
from utils.exp_metrics_plotter import MetricsPlotter
from utils.utils import set_settings
from utils.exp_config import get_config


class experiment:
    def __init__(self, config):
        self.config = config

    def load_data(self, config):
        if config.model == 'ours':
            all_x, all_y = get_financial_data('2020-07-13', '2025-03-8', config)
        else:
            all_x, all_y = get_ts(config.dataset, config)
        return all_x, all_y


# 数据集定义
class DataModule:
    def __init__(self, exper_type, config):
        self.config = config
        self.path = config.path
        self.x, self.y = exper_type.load_data(config)
        self.scaler = StandardScaler()
        if config.debug:
            self.x, self.y = self.x[:300], self.y[:300]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {np.max(self.y):.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )


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
        print(inputs.shape, label.shape)
        # break
    print('Done!')
