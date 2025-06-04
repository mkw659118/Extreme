# coding : utf-8
# Author : Yuxiang Zeng
from data_provider.exp_dataset import TensorDataset, TimeSeriesDataset
from data_provider.generate_financial import generate_data
from data_provider.get_financial import get_financial_data, multi_dataset
from data_provider.get_lottery import get_lottery
from data_provider.get_ts import get_ts
from utils.data_dataloader import get_dataloaders
from utils.data_spliter import get_split_dataset


def load_data(config):
    if config.dataset == 'financial':
        if config.multi_dataset:
            all_x, all_y, scaler = multi_dataset(config)
        else:
            all_x, all_y, scaler = get_financial_data(config.start_date, config.end_date, config.idx, config)
    elif config.dataset == 'weather':
        all_x, all_y, scaler = get_ts(config.dataset, config)
    elif config.dataset == 'lottery':
        all_x, all_y, scaler = get_lottery(config.dataset, config)
    return all_x, all_y, scaler

# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.x, self.y, self.scaler = load_data(config)
        if config.debug:
            self.x, self.y = self.x[:int(len(self.x) * 0.10)], self.y[:int(len(self.x) * 0.10)]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        if config.dataset == 'financial':
            return (
                TensorDataset(train_x, train_y, 'train', config),
                TensorDataset(valid_x, valid_y, 'valid', config),
                TensorDataset(test_x, test_y, 'test', config)
            )
        else:
            return (
                TimeSeriesDataset(train_x, train_y, 'train', config),
                TimeSeriesDataset(valid_x, valid_y, 'valid', config),
                TimeSeriesDataset(test_x, test_y, 'test', config)
            )

