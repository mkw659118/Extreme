# coding : utf-8
# Author : yuxiang Zeng

import numpy as np

def preprocess_data(x, y, config):
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    return x, y