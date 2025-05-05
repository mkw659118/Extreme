from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import os
import pickle
def get_each_cluster_group_idx():
    all_address = os.listdir('../datasets/financial/')
    print(len(all_address))

    minimal_length = 1e9
    maximum_length = -1
    all_length = []
    for i in range(len(all_address)):
        with open(os.path.join('../datasets/financial/', all_address[i]), 'rb') as f:
            now_df = pickle.load(f)
        minimal_length = min(minimal_length, len(now_df))
        maximum_length = max(maximum_length, len(now_df))
        all_length.append(len(now_df))
    all_length = np.array(all_length)
    minimal_threshold = int(np.percentile(all_length, q=10))
    maximum_threshold = int(np.percentile(all_length, q=99))
    train_length = int(maximum_length * 0.3)
    print(minimal_threshold, maximum_threshold)

    X = []
    all_func_code_list = []  # 用于保存所有被选中的 func_code（第一列）

    for i in range(len(all_address)):
        with open(os.path.join('../datasets/financial/', all_address[i]), 'rb') as f:
            now_df = pickle.load(f)
        if len(now_df) < train_length:
            continue
        selected_seq = now_df[:train_length, :]
        X.append(selected_seq)
        all_func_code_list.append(selected_seq[:, 0])  # 保留 func_code 列

    X = np.stack(X)
    all_func_code = np.stack(all_func_code_list)  # shape: [n_samples, train_length]
    X_features = X[:, :, -3:]  # 取最后三列作为特征
    print("X_features shape:", X_features.shape)  # (n_samples, train_length, 3)

    # 使用 DTW 的 KMeans 聚类
    model = TimeSeriesKMeans(
        n_clusters=20,
        metric="dtw",
        max_iter=10,
        n_jobs=12,
        random_state=42,
        verbose=True
    )
    labels = model.fit_predict(X_features)
    print("聚类标签:", labels)

    # 构造映射关系：取每个样本的第一个 func_code 作为代表（或者你也可以用 mode、mean 等）
    # 这里我们取第一个时间步的 func_code 作为该序列的 ID
    representative_func_codes = all_func_code[:, 0]  # shape: [n_samples, ]

    # 构建结果数组：每一行是 [func_code, label]
    mapping_array = np.column_stack((representative_func_codes, labels))

    # 保存到本地
    with open('../results/func_code_to_label.pkl', 'wb') as f:
        pickle.dump(mapping_array, f)

    print("映射表已保存至 '../results/func_code_to_label.pkl'")
    return mapping_array

if __name__ == '__main__':
    # mapping_array = get_each_cluster_group_idx()
    with open('../results/func_code_to_label.pkl', 'rb') as f:
        pickle.load(f)
    # print(mapping_array)
