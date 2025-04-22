import pickle
import os

import numpy as np
from utils.data_scaler import get_scaler

def main():
    all_fund = os.listdir('./datasets/financial/')
    for i in range(len(all_fund)):
        with open(f'./datasets/financial/{all_fund[0]}', 'rb') as f:
            df = pickle.load(f)
            print(df)
        break

    x = np.array(df)
    for i in range(3):
        x[:, -i] = x[:, -i].astype(np.float32)
        scaler = get_scaler(x[:, -i], config)
        x[:, -i] = scaler.transform(x[:, -i])
    print(x)
    # temp = x[:, -1].astype(np.float32)
    # x[:, -1] = (temp - scaler.y_mean) / scaler.y_std
    # print(df)
    # print(df.shape)


if __name__ == '__main__':
    from utils.exp_config import get_config
    config = get_config()
    main()

