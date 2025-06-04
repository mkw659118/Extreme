# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from einops import rearrange
class DataScalerStander:
    def __init__(self, y, config):
        self.config = config
        # 按 density 截取训练数据，然后 reshape 成二维
        train_data = y[:int(len(y) * self.config.density)].astype(np.float32)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)

    def transform(self, y):
        y = self.__check_input__(y)
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        y = self.__check_input__(y)
        return self.scaler.inverse_transform(y)
    
    def __check_input__(self, y):
        if isinstance(y, np.ndarray):
            y = y.astype(float)
        elif isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy().astype(float)
        
        if len(y.shape) == 3:
            y = y.reshape(y.shape[0], -1)
        return y 

class DataScalerMinMax:
    def __init__(self, y, config):
        self.config = config
        train_data = y[:int(len(y) * self.config.density)].astype(np.float32)
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)

    def transform(self, y):
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        return self.scaler.inverse_transform(y)
    


class GlobalStandardScaler:
    def __init__(self, y, config):
        self.config = config
        train_data = y[:int(len(y) * self.config.density)]
        train_data = self.__check_input__(train_data)
        self.mean = train_data.mean()
        self.std = train_data.std()
        if self.std == 0:
            self.std = 1

    def transform(self, x):
        x = self.__check_input__(x)
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        x = self.__check_input__(x)
        return x * self.std + self.mean

    def __check_input__(self, y):
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy().astype(float)
        elif isinstance(y, np.ndarray):
            y = y.astype(float)
        return y
    

def get_scaler(y, config):
    if config.scaler_method == 'stander':
        return DataScalerStander(y, config)
    elif config.scaler_method == 'minmax':
        return DataScalerMinMax(y, config)
    elif config.scaler_method == 'global':
        return GlobalStandardScaler(y, config)
    else:
        return NotImplementedError
    