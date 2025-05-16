# coding : utf-8
# Author : Yuxiang Zeng
from exp.exp_dataset import TensorDataset
from modules.load_data.get_financial import get_financial_data, multi_dataset
from modules.load_data.get_lottery import get_lottery
from modules.load_data.get_ts import get_ts
from utils.data_dataloader import get_dataloaders
from utils.data_spliter import get_split_dataset


def load_data(config):
    if config.dataset == 'financial':
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
            self.x, self.y = self.x[:300], self.y[:300]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)

        if config.dataset == 'financial' and config.multi_dataset:
            # self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.scaler = multi_dataset(config)
            self.x, self.y, self.scaler = multi_dataset(config)
            self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = get_split_dataset(self.x, self.y, config)
            self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
            config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, config):
        return (
            TensorDataset(train_x, train_y, 'train', config),
            TensorDataset(valid_x, valid_y, 'valid', config),
            TensorDataset(test_x, test_y, 'test', config)
        )

