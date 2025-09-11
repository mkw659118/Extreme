# coding : utf-8
# Author : Yuxiang Zeng
# 注意，这里的代码已经几乎完善，非必要不要改动（2025年06月22日15:57:27）
import os
import random
import platform
import numpy as np
import pandas as pd 
import multiprocessing
from torch.utils.data import DataLoader
from data_provider.data_control import get_dataset, load_data


# 数据集定义
class DataModule:
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.x, self.y, self.x_scaler, self.y_scaler = load_data(config)
        if config.debug:
            self.x, self.y = self.x[:int(len(self.x) * 0.10)], self.y[:int(len(self.x) * 0.10)]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self.get_split_dataset(self.x, self.y, config)
        self.train_set, self.valid_set, self.test_set = get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, config)
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataloaders(self.train_set, self.valid_set, self.test_set, config)
        config.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')
    
    def preprocess_data(self, x, y):
        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        return x, y
    
    def get_split_dataset(self, x, y, config):
        x, y = self.preprocess_data(x, y)
         # 新增：切换模式
        if config.split_mode == "ds":
            return ds_like_split_dataset(x, y, config)
         # 旧逻辑：比例切分
        train_ratio, valid_ratio, _ = parse_split_ratio(config.spliter_ratio)

        if config.use_train_size:
            train_size = int(config.train_size)
        else:
            train_size = int(len(x) * train_ratio)

        if config.eval_set:
            valid_size = int(len(x) * valid_ratio)
        else:
            valid_size = 0

        if config.classification:
            return get_train_valid_test_classification_dataset(x, y, train_size, valid_size, config)
        else:
            return get_train_valid_test_dataset(x, y, train_size, valid_size, config)
        

    def get_dataloaders(self, train_set, valid_set, test_set, config):

        if platform.system() == 'Linux' and 'ubuntu' in platform.version().lower():
            max_workers = multiprocessing.cpu_count() // 5
            # max_workers = 2
            prefetch_factor = 2
        else:
            max_workers = 0
            prefetch_factor = None

        train_loader = DataLoader(
            train_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda batch: train_set.custom_collate_fn(batch),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: valid_set.custom_collate_fn(batch),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config.bs,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: test_set.custom_collate_fn(batch),
            num_workers=max_workers,
            prefetch_factor=prefetch_factor
        )
        return train_loader, valid_loader, test_loader



def parse_split_ratio(ratio_str):
    # 解析如 '7:1:2' 的字符串为归一化比例 [0.7, 0.1, 0.2]
    parts = list(map(int, ratio_str.strip().split(':')))
    total = sum(parts)
    return [p / total for p in parts]


def get_train_valid_test_dataset(x, y, train_size, valid_size, config):
    if config.shuffle:
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]
    train_x = x[:train_size]
    train_y = y[:train_size]
    valid_x = x[train_size:train_size + valid_size]
    valid_y = y[train_size:train_size + valid_size]
    test_x = x[train_size + valid_size:]
    test_y = y[train_size + valid_size:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_train_valid_test_classification_dataset(x, y, train_size, valid_size):
    from collections import defaultdict
    import random
    class_data = defaultdict(list)
    for now_x, now_label in zip(x, y):
        class_data[now_label].append(now_x)
    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x, test_y = [], []
    for label, now_x in class_data.items():
        random.shuffle(now_x)
        train_x.extend(now_x[:train_size])
        train_y.extend([label] * len(now_x[:train_size]))
        valid_x.extend(now_x[train_size:train_size + valid_size])
        valid_y.extend([label] * len(now_x[train_size:train_size + valid_size]))
        test_x.extend(now_x[train_size + valid_size:])
        test_y.extend([label] * len(now_x[train_size + valid_size:]))
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def ds_like_split_dataset(x, y, config):

    """
    DS风格切分 (带过采样/极值/邻域屏蔽):
    - start_point / train_end / test_start / test_end 都是字符串时间戳
    - 训练集: 随机采样 + 极值过采样
    - 验证集: 随机 anchor + 邻域屏蔽
    - 测试集: 滚动窗口
    """
    
    # ==== 读 CSV 获取时间列 ====
    csv_path = os.path.join(config.path, f"{config.reservoir_sensor}.tsv")
    df = pd.read_csv(csv_path, sep = '\t')
    time_col = df.columns[0]
    time_series = pd.to_datetime(df[time_col])

    def to_idx(t_str):
        idxs = df.index[time_series == pd.to_datetime(t_str)].tolist()
        if not idxs:
            raise ValueError(f"时间点 {t_str} 不在数据中")
        return idxs[0]

    # ==== 转 index ====
    start_point_idx   = to_idx(config.start_point)
    train_end_idx  = to_idx(config.train_end)
    test_start_idx = to_idx(config.test_start)
    test_end_idx = to_idx(config.test_end)

    # ==== 长度参数 ====
    L_in  = config.seq_len
    L_out = config.pred_len
    roll  = config.roll

    # ==== 候选池 ====
    cand_lo = start_point_idx + L_in
    cand_hi = train_end_idx - L_out
    assert cand_hi > cand_lo, "训练候选区间不足，请检查 start/train_end 与 seq_len/pred_len"

    # ==== extreme 阈值 ====
    y_series = y[:, 0] if y.ndim == 2 else y
    ref = y_series[cand_lo: cand_hi + L_out]
    q_low, q_high = np.nanpercentile(ref, [10, 90])

    def is_extreme(seg):
        return (np.nanmax(seg) >= q_high) or (np.nanmin(seg) <= q_low)

    # ==== 验证集 ====
    random.seed(config.val_seed)
    val_points = []
    tag = np.zeros(len(x), dtype=np.int8)
    near_len = L_out
    while len(val_points) < config.val_size:
        i = random.randint(cand_lo, cand_hi)
        win_x = x[i - L_in:i + L_out]
        win_y = y[i - L_in:i + L_out]
        if np.isnan(win_x).any() or np.isnan(win_y).any():
            continue
        if tag[i] != 0:
            continue
        tag[i] = 2  # 验证 anchor
        tag[max(i - near_len, 0):min(i + near_len + 1, len(x))] = 3  # 屏蔽邻域
        val_points.append(i)

    valid_x = np.array([x[i - L_in:i] for i in val_points], dtype=np.float32)
    valid_y = np.array([y[i:i + L_out] for i in val_points], dtype=np.float32)

    # ==== 训练集 (含过采样) ====
    random.seed(config.train_seed)
    train_x, train_y = [], []
    train_volume = config.train_volume
    oversampling_pct = config.oversampling
    os_steps = config.os_s
    os_stride = config.os_v
    n_over_keep = train_volume * (oversampling_pct / 100)
    n_over_now = 0

    while len(train_x) < train_volume:
        i = random.randint(cand_lo, cand_hi)
        if tag[i] in (2, 3, 4):
            continue
        wp_x = x[i - L_in:i]
        wf_y = y[i:i + L_out]
        if np.isnan(wp_x).any() or np.isnan(wf_y).any():
            continue

        # 极值过采样
        if n_over_now < n_over_keep and is_extreme(wf_y):
            ii = i
            for _ in range(max(os_steps, 0)):
                ii = ii + max(os_stride, 1)
                if ii >= cand_hi:
                    break
                if tag[ii] in (2, 3, 4):
                    continue
                wp_x2 = x[ii - L_in:ii]
                wf_y2 = y[ii:ii + L_out]
                if np.isnan(wp_x2).any() or np.isnan(wf_y2).any():
                    continue
                train_x.append(wp_x2)
                train_y.append(wf_y2)
                tag[ii] = 4
                n_over_now += 1
                if len(train_x) >= train_volume:
                    break

        train_x.append(wp_x)
        train_y.append(wf_y)
        tag[i] = 4

    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.float32)

    # ==== 测试集 ====
    test_x, test_y = [], []
    for s in range(test_start_idx, test_end_idx - L_out + 1, roll):
        wp_x = x[s - L_in:s]
        wf_y = y[s:s + L_out]
        if np.isnan(wp_x).any() or np.isnan(wf_y).any():
            continue
        test_x.append(wp_x)
        test_y.append(wf_y)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)
    print(f"[ds_like_split_dataset] train_x={train_x.shape}, valid_x={valid_x.shape}, test_x={test_x.shape}")


    return train_x, train_y, valid_x, valid_y, test_x, test_y
