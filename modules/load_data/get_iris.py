# coding : utf-8
# Author : yuxiang Zeng
import pandas as pd
import numpy as np
import sklearn.preprocessing
def get_iris():
    df = pd.read_csv('./datasets/iris_synthetic_data.csv').to_numpy()
    all_x = df[:, :-1]
    all_y = df[:, -1]
    feature_scaler = sklearn.preprocessing.MinMaxScaler()
    label_encoder = sklearn.preprocessing.LabelEncoder()
    all_x = feature_scaler.fit_transform(all_x)
    all_y = label_encoder.fit_transform(all_y)
    return all_x, all_y
