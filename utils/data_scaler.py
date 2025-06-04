# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import torch 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    

def get_scaler(y, config):
    if config.scaler_method == 'stander':
        return DataScalerStander(y, config)
    elif config.scaler_method == 'minmax':
        return DataScalerMinMax(y, config)
    else:
        return NotImplementedError