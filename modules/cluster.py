import numpy as np
import os
import pickle
from random import sample

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax

def dtw_kmeans(X, n_clusters=10, max_iter=10, random_state=42, verbose=True):
    # 先对所有时间序列进行归一化处理
    X_scaled = TimeSeriesScalerMinMax().fit_transform(X)

    # 使用 tslearn 的 DTW 距离版本的 KMeans
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=max_iter,
        verbose=verbose,
        random_state=random_state,
        n_jobs=14
    )
    labels = model.fit_predict(X_scaled)
    return labels

def get_each_cluster_group_idx():
    all_address = os.listdir('../datasets/financial/')
    print(len(all_address))

    all_length = []
    for file in all_address:
        with open(os.path.join('../datasets/financial/', file), 'rb') as f:
            now_df = pickle.load(f)
        all_length.append(len(now_df))

    all_length = np.array(all_length)
    minimal_threshold = int(np.percentile(all_length, 10))
    maximum_threshold = int(np.percentile(all_length, 99))
    train_length = int(np.max(all_length) * 0.3)
    print(minimal_threshold, maximum_threshold)

    X = []
    all_func_code_list = []
    for file in all_address:
        with open(os.path.join('../datasets/financial/', file), 'rb') as f:
            now_df = pickle.load(f)
        if len(now_df) < train_length:
            continue
        selected_seq = now_df[:train_length, :]
        X.append(selected_seq[:, -3:])  # 只取特征部分
        all_func_code_list.append(selected_seq[:, 0])  # 保存 func_code

    X = np.array(X, dtype=object)  # list of (train_length, 3)
    all_func_code = np.stack(all_func_code_list)
    print("X shape:", len(X), "with each of shape:", X[0].shape)

    labels = dtw_kmeans(X, n_clusters=10, max_iter=10, verbose=True)
    print("聚类标签:", labels)

    representative_func_codes = all_func_code[:, 0]
    mapping_array = np.column_stack((representative_func_codes, labels))

    os.makedirs('../results', exist_ok=True)
    with open('../results/func_code_to_label.pkl', 'wb') as f:
        pickle.dump(mapping_array, f)

    print("映射表已保存至 '../results/func_code_to_label.pkl'")
    return mapping_array

if __name__ == '__main__':
    mapping_array = get_each_cluster_group_idx()

    with open('../results/func_code_to_label.pkl', 'rb') as f:
        mapping_array = pickle.load(f)

    dic = {}
    for i in range(len(mapping_array)):
        label = mapping_array[i][1]
        dic[label] = dic.get(label, 0) + 1
    print(dic)