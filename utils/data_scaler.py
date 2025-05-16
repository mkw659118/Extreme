# coding : utf-8
# Author : yuxiang Zeng
import numpy as np

class DataScalerStander:
    def __init__(self, y, config):
        self.config = config
        # 根据训练集的数值进行归一化
        scaler = y[:int(len(y) * self.config.density)].astype(np.float32)
        self.y_mean = scaler.mean()
        self.y_std = scaler.std() + 1e-9
        # print(self.y_mean, self.y_std)

    def transform(self, y):
        return (y - self.y_mean) / self.y_std if self.y_std != 0 else y - self.y_mean

    def inverse_transform(self, y):
        return y * self.y_std + self.y_mean if self.y_std != 0 else y + self.y_mean

class DataScalerMinMax:
    def __init__(self, y, config):
        self.config = config
        # 根据训练集的数值进行归一化
        scaler = y[:int(len(y) * self.config.density)].astype(np.float32) * 1.2
        # scaler = y
        self.y_min = scaler.min()
        self.y_max = scaler.max()

    def transform(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min) if self.y_max != self.y_min else y - self.y_min

    def inverse_transform(self, y):
        return y * (self.y_max - self.y_min) + self.y_min if self.y_max != self.y_min else y + self.y_min


def get_scaler(y, config):
    if config.scaler_method == 'stander':
        return DataScalerStander(y, config)
    elif config.scaler_method == 'minmax':
        return DataScalerMinMax(y, config)
    else:
        return NotImplementedError