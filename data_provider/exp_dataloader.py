# coding : utf-8
# Author : Yuxiang Zeng
from data_provider.exp_dataset import TensorDataset, TimeSeriesDataset
from data_provider.generate_financial import generate_data
from data_provider.get_financial import get_financial_data, multi_dataset
from data_provider.get_ts import get_ts
from data_provider.data_dataloader import get_dataloaders
from data_provider.data_spliter import get_split_dataset


def load_data(config):
    if config.dataset == 'financial':
        if config.multi_dataset:
            x, y, x_scaler, y_scaler = multi_dataset(config)
        else:
            x, y, x_scaler, y_scaler = get_financial_data(config.start_date, config.end_date, config.idx, config)
    elif config.dataset == 'weather':
        x, y, x_scaler, y_scaler = get_ts(config.dataset, config)
    return x, y, x_scaler, y_scaler

# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.x, self.y, self.x_scaler, self.y_scaler = load_data(config)
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

