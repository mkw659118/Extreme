# coding : utf-8
# Author : yuxiang Zeng
import numpy as np

class DataScaler:
    def __init__(self, y, config):
        self.config = config
        # 根据训练集的数值进行归一化
        scaler = y[:int(len(y) * self.config.density)].astype(np.float32)
        self.y_mean = scaler.mean()
        self.y_std = scaler.std()
        print(self.y_mean, self.y_std)

    def transform(self, y):
        return (y - self.y_mean) / self.y_std if self.y_std != 0 else y - self.y_mean

    def inverse_transform(self, y):
        return y * self.y_std + self.y_mean if self.y_std != 0 else y + self.y_mean

def get_scaler(y, config):
    return DataScaler(y, config)