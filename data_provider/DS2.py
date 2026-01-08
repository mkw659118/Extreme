# Author  : mkw (modified by assistant)
# Time    : 2025/09/30
# Desc    : DS with chronological split (val = last val_size windows), no leakage in stats/GMMs

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from utils.utils2 import (
    diff_order_1,
    gen_month_tag,
    gen_time_feature,
    cos_date,
    sin_date,
    r_log_std_normalization,
    r_log_std_normalization_1,
)

from data_provider.data_getitem import TimeSeriesDataset


class DS2:
    """数据处理类：支持按时间线段划分（验证集为末尾一段），并避免统计量/GMM泄漏。"""

    def __init__(self, config, trainX):
        self.config = config
        self.trainX = trainX

        # 统计参数
        self.mean = 0
        self.std = 0
        self.mini = 0

        # 原始/时间字段
        self.sensor_data = []
        self.data = []
        self.data_time = []

        # 差分（可选）
        self.diff_data = []

        # 归一化与特征
        self.sensor_data_norm = []     # 归一化后的 value（1D）
        self.sensor_data_norm1 = []    # 组装后的输入特征（二维）

        # 时间特征
        self.month_tag = []  # 只存月份标签，不再被覆盖
        self.month = []
        self.day = []
        self.hour = []

        # 划分信息（关键）
        self.val_starts = None
        self.val_centers = None
        self.val_start_i = None  # 验证集最早窗口起点 i（train_cut）

        # 阈值/模型
        self.gm3 = GaussianMixture(n_components=3)
        self.gmm0 = GaussianMixture(n_components=3)
        self.gmm = GaussianMixture(n_components=3)  # 窗口级 GMM（训练/验证/测试时用）
        self.thre1 = 0
        self.thre2 = 0
        self.gmm_l = self.config.pred_len

        # 采样与数据长度
        self.oversampling = int(getattr(config, "oversampling", 0))
        self.iterval = int(getattr(config, "os_v", 1))
        self.os_h = int(getattr(config, "os_s", 0))
        self.os_l = int(getattr(config, "os_v", 0))  # 原代码如此写；保留
        self.seq_len = int(self.config.seq_len)
        self.pred_len = int(self.config.pred_len)
        self.lens = self.seq_len + self.pred_len + 1
        self.batch_size = int(self.config.bs)

        # DataLoader
        self.val_data_loader = []
        self.train_data_loader = []
        self.test_data_loader = []

        # 时间范围
        self.test_start_time = self.config.test_start
        self.test_end_time = self.config.test_end

        # 目录
        self.expr_dir = os.path.join(self.config.outf, self.config.reservoir_sensor, "train")
        os.makedirs(self.expr_dir, exist_ok=True)

        # 读取并预处理（含：确定 val_starts；仅用 train_cut 拟合统计量/gm3/gmm0；生成特征）
        self.read_dataset()

        # 保存 mean/std
        norm = [self.get_mean(), self.get_std()]
        np.savetxt(os.path.join(self.expr_dir, "Norm.txt"), norm)

        # 推理滚动间隔
        self.roll = 8

        if self.config.mode == "train":
            # 临时训练窗口级 GMM（只用训练段，供 val 生成窗口级概率特征使用）
            self.train_temp_gmm()

            # 生成 val/train dataloader（val 固定尾段；train 只用 val 前）
            self.val_dataloader()
            self.train_dataloader()

            # 刷新到包含测试期的全序列特征（用训练期的 mean/std + gm3/gmm0）
            self.refresh_dataset(trainX)

            print("[TEST] 构建测试集...")
            self.gen_test_data()

            # 打印维度检查
            arr = np.asarray(self.sensor_data_norm1)
            print("sensor_data_norm1 shape:", arr.shape)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                print("样本第0列（异常概率）:", arr[:10, 0])
                print("样本第1列（核心数值）:", arr[:10, 1])
            print("训练 mean:", self.mean, "训练 std:", self.std)

    # ----------------------- getter -----------------------
    def get_mean(self): return self.mean
    def get_std(self): return self.std
    def get_val_data_loader(self): return self.val_data_loader
    def get_train_data_loader(self): return self.train_data_loader
    def get_sensor_data_norm1(self): return self.sensor_data_norm1

    # ----------------------- helpers -----------------------
    @staticmethod
    def _prefix_nan_any(x_2d: np.ndarray) -> np.ndarray:
        """返回 prefix 累积：invalid_row[t]=该行是否含 NaN。prefix[k]=sum(invalid_row[:k])"""
        invalid_row = np.isnan(x_2d).any(axis=1).astype(np.int32)
        prefix = np.zeros(len(invalid_row) + 1, dtype=np.int32)
        prefix[1:] = np.cumsum(invalid_row)
        return prefix

    def _month_ok_val(self, v: int) -> bool:
        # 保留你原始验证筛选逻辑
        a1, a2 = 0, -13
        return (v <= a1) or (a2 < v < 0) or (2 <= v <= 3)

    def _month_ok_train(self, v: int) -> bool:
        # 保留你原始训练筛选逻辑
        a1, a2 = 0, -13
        return (v <= a1) or (a2 < v < 0)

    @staticmethod
    def _reorder_probs_by_weights(prob: np.ndarray, weights: np.ndarray, mode: str) -> np.ndarray:
        """
        复刻你原来的 reorder 逻辑：
        - mode="gmm0": order1=argmax(weights), order2=argmin(weights), 剩余为 order3
        - mode="gmm":  order1=argmin(weights), order2=argmax(weights), 剩余为 order3
        """
        weights = np.asarray(weights).reshape(-1)
        if mode == "gmm0":
            order1 = int(np.argmax(weights))
            order2 = int(np.argmin(weights))
        else:
            order1 = int(np.argmin(weights))
            order2 = int(np.argmax(weights))

        order3 = [i for i in range(3) if i != order1 and i != order2][0]
        d0 = prob[:, order1:order1 + 1]
        d1 = prob[:, order2:order2 + 1]
        d2 = prob[:, order3:order3 + 1]
        return np.concatenate([d0, d1, d2], axis=1)

    # ----------------------- core pipeline -----------------------
    def read_dataset(self):
        """
        读取训练期数据（start_point ~ train_end），并做：
        1) 生成 month_tag + time_feature
        2) 构造候选窗口，按时间取最后 val_size 个作为验证集
        3) 定义 train_cut = val_start_i；仅用 train_cut 前拟合 mean/std、gm3、gmm0
        4) 用训练期拟合的统计量与 gm3/gmm0 为整段训练期生成输入特征 sensor_data_norm1（5维）
        """
        # 1) 切训练期
        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]
        train_end_rel = self.trainX[self.trainX["datetime"] == self.config.train_end].index.values[0] - start_num
        self.sensor_data = self.trainX[start_num: start_num + train_end_rel + 1]

        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        T = len(self.data)

        # 差分（可选）
        self.diff_data = diff_order_1(self.data)

        # 2) 时间标签（只做月份标签，不污染）
        self.month_tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        # 3) 构造候选窗口（仅用 NaN 与月份条件决定验证集，不依赖 GMM 特征）
        #    这里用“仅 value 一列”做 NaN 判定即可
        value_col = self.data.reshape(-1, 1)
        prefix = self._prefix_nan_any(value_col)

        lens = self.lens
        i_min = self.pred_len
        i_max = T - lens - 1
        if i_max <= i_min:
            raise ValueError(f"数据长度不足：T={T}, lens={lens}")

        starts = np.arange(i_min, i_max + 1, dtype=np.int32)
        win_bad = prefix[starts + lens] - prefix[starts]
        starts = starts[win_bad == 0]
        if len(starts) == 0:
            raise ValueError("找不到任何窗口无 NaN 的候选样本。")

        centers = starts + self.seq_len
        month_vals = np.asarray(self.month_tag)[centers].astype(np.int32)
        ok = np.array([self._month_ok_val(int(v)) for v in month_vals], dtype=bool)
        starts = starts[ok]
        centers = centers[ok]
        if len(starts) == 0:
            raise ValueError("候选窗口均不满足月份筛选条件，请检查 gen_month_tag 或放宽条件。")

        # 按时间排序取最后 val_size 个
        order = np.argsort(centers)
        starts = starts[order]
        centers = centers[order]

        k = int(self.config.val_size)
        if k > len(starts):
            print(f"[VAL][WARN] val_size={k} > 候选数={len(starts)}，将使用全部候选。")
            k = len(starts)

        self.val_starts = starts[-k:]
        self.val_centers = centers[-k:]
        self.val_start_i = int(self.val_starts[0])  # train_cut

        print(f"[SPLIT] 训练期长度 T={T}")
        print(f"[SPLIT] val_size={k}, val_start_i(train_cut)={self.val_start_i}, "
              f"val_center_range=[{int(self.val_centers[0])}, {int(self.val_centers[-1])}]")

        # 4) 仅用 train_cut 前拟合归一化参数
        train_cut = self.val_start_i
        train_slice = self.data[:train_cut]

        # r_log_std_normalization 内部会处理 NaN（按你原工程实现），这里直接调用得到 mean/std/mini
        _, self.mean, self.std, self.mini = r_log_std_normalization(train_slice)

        # 用训练参数对整个训练期做同分布归一化
        self.sensor_data_norm = r_log_std_normalization_1(self.data, self.mean, self.std)

        # 5) 拟合 gm3（仅训练段），并生成 outlier 概率特征（整段）
        clean_train = self.sensor_data_norm[:train_cut]
        clean_train = clean_train[~np.isnan(clean_train)]
        if len(clean_train) < 10:
            raise ValueError("训练段有效样本太少，无法拟合 gm3。")
        self.gm3.fit(np.array(clean_train, np.float32).reshape(-1, 1))
        torch.save(self.gm3, os.path.join(self.expr_dir, "GM3.pt"))

        gm_means = np.squeeze(self.gm3.means_)
        z0, z1, z2 = float(np.min(gm_means)), float(np.median(gm_means)), float(np.max(gm_means))
        self.thre1 = (z0 + z1) / 2.0
        self.thre2 = (z1 + z2) / 2.0
        print("[GM3] means:", gm_means, "thre1:", self.thre1, "thre2:", self.thre2)

        # 计算 outlier-like prob：对全段非 NaN 预测再恢复长度
        full_clean = self.sensor_data_norm[~np.isnan(self.sensor_data_norm)].astype(np.float32).reshape(-1, 1)
        data_prob3 = self.gm3.predict_proba(full_clean)
        weights3 = self.gm3.weights_.reshape(-1)

        prob_in = data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2]
        prob_out = (1.0 - prob_in).reshape(-1, 1).astype(np.float32)

        # 恢复到原长度（NaN 位置保持 NaN）
        prob_out_full = np.full((T, 1), np.nan, dtype=np.float32)
        j = 0
        for i in range(T):
            if not np.isnan(self.sensor_data_norm[i]):
                prob_out_full[i, 0] = prob_out[j, 0]
                j += 1

        # 组装 base 特征：保证列顺序为 [outlier_prob, value]
        value_full = np.array(self.sensor_data_norm, np.float32).reshape(T, 1)
        self.sensor_data_norm1 = np.concatenate([prob_out_full, value_full], axis=1)

        # 6) 拟合 gmm0（仅训练段），并生成点级 3 概率（整段）
        #    这里限制随机采样只从训练段 [0, train_cut) 取
        gmm0_samples = int(getattr(self.config, "gmm0_samples", 200000))
        upper = max(1, train_cut - self.gmm_l - 1)

        series = []
        random.seed(int(getattr(self.config, "val_seed", 0)))
        tries = 0
        max_tries = gmm0_samples * 2

        while len(series) < gmm0_samples and tries < max_tries:
            tries += 1
            g0 = random.randint(0, upper)
            v = self.sensor_data_norm[g0]
            if not np.isnan(v):
                series.append([v])

        if len(series) < 50:
            # 训练段太短或 NaN 太多，兜底用全部训练段有效点
            series = clean_train.astype(np.float32).reshape(-1, 1).tolist()

        self.gmm0.fit(np.array(series, dtype=np.float32).reshape(-1, 1))
        torch.save(self.gmm0, os.path.join(self.expr_dir, "GMM0.pt"))

        # 计算全段点级概率并恢复
        full_clean = self.sensor_data_norm[~np.isnan(self.sensor_data_norm)].astype(np.float32).reshape(-1, 1)
        prob0 = self.gmm0.predict_proba(full_clean)
        prob0 = self._reorder_probs_by_weights(prob0, self.gmm0.weights_, mode="gmm0")  # [N,3]

        prob0_full = np.zeros((T, 3), dtype=np.float32)
        j = 0
        for i in range(T):
            if not np.isnan(self.sensor_data_norm[i]):
                prob0_full[i, :] = prob0[j, :]
                j += 1
            else:
                prob0_full[i, :] = 0.0

        # 追加 3 维点级概率 => sensor_data_norm1 变为 5 维
        self.sensor_data_norm1 = np.concatenate([self.sensor_data_norm1, prob0_full], axis=1)

        print("[FEATURE] train-period sensor_data_norm1 shape:", np.asarray(self.sensor_data_norm1).shape)

    def train_temp_gmm(self):
        """
        临时训练窗口级 GMM（仅用训练段，不泄漏到验证段）
        用于 val_dataloader 生成窗口级 3 概率特征。
        """
        train_cut = int(self.val_start_i)
        if train_cut <= self.seq_len + self.pred_len + 2:
            raise ValueError("训练段太短，无法训练临时窗口级 GMM。")

        x = np.asarray(self.sensor_data_norm1[:train_cut], dtype=np.float32)
        x = x[~np.isnan(x).any(axis=1)]

        window_size = self.gmm_l
        if len(x) < window_size + 2:
            raise ValueError("训练段有效样本不足，无法训练临时窗口级 GMM。")

        # 用滑动窗口取最多 1000 个样本（快且稳定）
        max_possible = len(x) - window_size + 1
        n_samples = min(max_possible, 1000)
        samples = []
        for i in range(n_samples):
            # 取 value 列（第 1 列）作为窗口输入
            w = x[i:i + window_size, 1:2].flatten()
            samples.append(w)
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) < 2:
            samples = np.repeat(samples, 2, axis=0)

        self.gmm = GaussianMixture(n_components=3)
        self.gmm.fit(samples)
        torch.save(self.gmm, os.path.join(self.expr_dir, "GMM.pt"))
        print(f"[TEMP GMM] fitted with {len(samples)} samples, window={window_size}")

    def val_dataloader(self):
        """
        验证集：固定为训练期候选窗口中“最后 val_size 个”窗口（不随机）。
        并追加窗口级 GMM 3 概率（来自 train_temp_gmm）。
        """
        print("Begin to generate val_dataloader (tail split)...")

        if self.val_starts is None or self.val_centers is None:
            raise RuntimeError("val_starts 未初始化，请先运行 read_dataset().")

        DATA, Label = [], []
        self.val_points = []

        for i in self.val_starts:
            i = int(i)
            j = i + self.seq_len
            point = self.data_time[j]
            self.val_points.append([point])

            data0 = np.array(self.sensor_data_norm1[i: i + self.seq_len], dtype=np.float32).reshape(self.seq_len, -1)

            label00 = np.array(self.sensor_data_norm[i + self.seq_len: i + self.seq_len + self.pred_len], dtype=np.float32)
            label0 = [[ff] for ff in label00]

            b = i + self.seq_len
            e = i + self.seq_len + self.pred_len

            label2 = [[ff] for ff in cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])]
            label3 = [[ff] for ff in sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])]

            label4 = np.array(self.data[i + self.seq_len - 1: i + self.seq_len + self.pred_len - 1], dtype=np.float32).reshape(-1, 1)
            label5 = np.array(self.data[i + self.seq_len: i + self.seq_len + self.pred_len], dtype=np.float32).reshape(-1, 1)

            label = np.concatenate((label0, label2), 1)
            label = np.concatenate((label, label3), 1)
            label = np.concatenate((label, label4), 1)
            label = np.concatenate((label, label5), 1)

            DATA.append(data0)
            Label.append(label)

        self.DATA_val = DATA
        self.Label_val = Label

        # 追加窗口级 GMM 概率特征（3维，重复到 seq_len）
        self.gmm = torch.load(os.path.join(self.expr_dir, "GMM.pt"), weights_only=False)
        xx = np.array(self.DATA_val, np.float32)

        gmm_prob30 = self.gmm.predict_proba(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        gmm_prob3 = self._reorder_probs_by_weights(gmm_prob30, self.gmm.weights_, mode="gmm")  # [B,3]

        prob0 = gmm_prob3[:, 0:1].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob1 = gmm_prob3[:, 1:2].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob2 = gmm_prob3[:, 2:3].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob = np.concatenate((prob0, prob1, prob2), axis=2)

        DATA = np.concatenate((DATA, prob), axis=2)

        print("Validation DATA shape:", np.array(DATA).shape)
        print("Validation Label shape:", np.array(Label).shape)

        dataset1 = TimeSeriesDataset(DATA, self.Label_val, self.config)
        self.val_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )

        # 保存验证集时间戳
        self.config.name = "%s" % (self.config.data_model)
        val_dir = os.path.join(self.config.outf, self.config.name, "val")
        os.makedirs(val_dir, exist_ok=True)
        file_name = os.path.join(val_dir, "validation_timestamps_24avg.tsv")
        pd.DataFrame(self.val_points, columns=["Hold Out Start"]).to_csv(file_name, sep="\t")
        print("val set saved to:", file_name)

    def train_dataloader(self):
        """
        训练集：只从 val_start_i 之前采样，且训练窗口需满足：
          start + seq_len + pred_len <= val_start_i
        避免与验证段重叠。
        """
        print("Begin to generate train_dataloader (pre-val only)...")

        T = len(self.data)
        lens = self.lens
        train_cut = int(self.val_start_i)
        gap_len = self.seq_len + self.pred_len

        # 枚举训练候选窗口：窗口无 NaN + 月份条件 + 严格落在 val_start_i 之前
        x_all = np.asarray(self.sensor_data_norm1, dtype=np.float32)
        prefix = self._prefix_nan_any(x_all)

        i_min = self.pred_len * 4
        i_max = min(T - lens - 1, train_cut - gap_len - 1)
        if i_max <= i_min:
            raise ValueError("切分后训练候选为空：请减小 val_size 或缩短 seq_len/pred_len。")

        starts = np.arange(i_min, i_max + 1, dtype=np.int32)
        win_bad = prefix[starts + lens] - prefix[starts]
        starts = starts[win_bad == 0]

        centers = starts + self.seq_len
        month_vals = np.asarray(self.month_tag)[centers].astype(np.int32)
        ok = np.array([self._month_ok_train(int(v)) for v in month_vals], dtype=bool)
        starts = starts[ok]

        # 严格不重叠
        starts = starts[starts + gap_len <= train_cut]

        if len(starts) == 0:
            raise ValueError("训练候选为空：约束过强或数据缺失过多。")

        print(f"[TRAIN] candidates={len(starts)}, target train_volume={self.config.train_volume}")

        # 过采样：先找极值窗口集合
        extreme = []
        for i in starts:
            b = int(i + self.seq_len)
            e = b + self.pred_len
            pre1 = np.array(self.sensor_data_norm[b:e], dtype=np.float32)
            if np.max(pre1) > self.thre2 or np.min(pre1) < self.thre1:
                extreme.append(int(i))
        extreme = np.asarray(extreme, dtype=np.int32)
        print(f"[TRAIN] extreme_candidates={len(extreme)}")

        train_volume = int(self.config.train_volume)
        n_over = int(train_volume * (self.oversampling / 100.0))
        n_over = max(0, min(n_over, train_volume))
        n_norm = train_volume - n_over

        rng = np.random.default_rng(int(getattr(self.config, "train_seed", 0)))

        selected = []
        if n_over > 0:
            if len(extreme) > 0:
                sel_over = rng.choice(extreme, size=n_over, replace=(len(extreme) < n_over))
            else:
                sel_over = rng.choice(starts, size=n_over, replace=(len(starts) < n_over))
            selected.extend([int(x) for x in sel_over])

        if n_norm > 0:
            sel_norm = rng.choice(starts, size=n_norm, replace=(len(starts) < n_norm))
            selected.extend([int(x) for x in sel_norm])

        rng.shuffle(selected)

        DATA, Label = [], []
        for i in selected:
            i = int(i)
            data0 = np.array(self.sensor_data_norm1[i: i + self.seq_len], dtype=np.float32).reshape(self.seq_len, -1)

            label00 = np.array(self.sensor_data_norm[i + self.seq_len: i + self.seq_len + self.pred_len], dtype=np.float32)
            label0 = [[ff] for ff in label00]

            b = i + self.seq_len
            e = i + self.seq_len + self.pred_len

            label2 = [[ff] for ff in cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])]
            label3 = [[ff] for ff in sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])]

            label4 = np.array(self.data[i + self.seq_len - 1: i + self.seq_len + self.pred_len - 1], dtype=np.float32).reshape(-1, 1)
            label5 = np.array(self.data[i + self.seq_len: i + self.seq_len + self.pred_len], dtype=np.float32).reshape(-1, 1)

            label = np.concatenate((label0, label2), 1)
            label = np.concatenate((label, label3), 1)
            label = np.concatenate((label, label4), 1)
            label = np.concatenate((label, label5), 1)

            DATA.append(data0)
            Label.append(label)

        if len(DATA) == 0:
            raise ValueError("最终训练 DATA 为空：请检查 NaN/筛选条件/val_size。")

        self.DATA = DATA
        self.Label = Label

        # 训练窗口级 GMM（用于后续 test/val 的窗口概率特征；这一步只用训练窗口，不泄漏）
        self.gmm = GaussianMixture(n_components=3)
        xx = np.array(self.DATA, np.float32)
        self.gmm.fit(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        torch.save(self.gmm, os.path.join(self.expr_dir, "GMM.pt"))
        print("[GMM] window-level weights:", self.gmm.weights_)

        # 生成训练集窗口级概率，并追加到 DATA => 特征维度变为 8（对齐模型切片）
        gmm_prob30 = self.gmm.predict_proba(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        gmm_prob3 = self._reorder_probs_by_weights(gmm_prob30, self.gmm.weights_, mode="gmm")

        prob0 = gmm_prob3[:, 0:1].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob1 = gmm_prob3[:, 1:2].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob2 = gmm_prob3[:, 2:3].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob = np.concatenate((prob0, prob1, prob2), axis=2)

        DATA = np.concatenate((DATA, prob), axis=2)

        print("Train DATA shape:", np.array(DATA).shape)
        print("Train Label shape:", np.array(Label).shape)
        print("最终训练样本数:", len(DATA))

        dataset1 = TimeSeriesDataset(DATA, self.Label, self.config)
        self.train_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )

    # ----------------------- refresh/test -----------------------
    def refresh_dataset(self, trainX):
        """
        刷新到包含测试期的更长序列：
        - 使用训练期拟合的 mean/std
        - 使用训练期拟合的 gm3/gmm0（避免泄漏）
        """
        print("刷新数据集********************")
        self.trainX = trainX

        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]
        k = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
        self.sensor_data = self.trainX[start_num:k]

        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))

        # 时间标签
        self.month_tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        # 归一化（用训练期参数）
        self.sensor_data_norm = r_log_std_normalization_1(self.data, self.mean, self.std)

        T = len(self.sensor_data_norm)

        # 加载/使用训练期 gm3、gmm0
        gm3_path = os.path.join(self.expr_dir, "GM3.pt")
        gmm0_path = os.path.join(self.expr_dir, "GMM0.pt")
        self.gm3 = torch.load(gm3_path, weights_only=False) if os.path.exists(gm3_path) else self.gm3
        self.gmm0 = torch.load(gmm0_path, weights_only=False) if os.path.exists(gmm0_path) else self.gmm0

        # outlier prob
        full_clean = self.sensor_data_norm[~np.isnan(self.sensor_data_norm)].astype(np.float32).reshape(-1, 1)
        data_prob3 = self.gm3.predict_proba(full_clean)
        weights3 = self.gm3.weights_.reshape(-1)

        prob_in = data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2]
        prob_out = (1.0 - prob_in).reshape(-1, 1).astype(np.float32)

        prob_out_full = np.full((T, 1), np.nan, dtype=np.float32)
        j = 0
        for i in range(T):
            if not np.isnan(self.sensor_data_norm[i]):
                prob_out_full[i, 0] = prob_out[j, 0]
                j += 1

        value_full = np.array(self.sensor_data_norm, np.float32).reshape(T, 1)
        feat = np.concatenate([prob_out_full, value_full], axis=1)

        # gmm0 prob
        prob0 = self.gmm0.predict_proba(full_clean)
        prob0 = self._reorder_probs_by_weights(prob0, self.gmm0.weights_, mode="gmm0")

        prob0_full = np.zeros((T, 3), dtype=np.float32)
        j = 0
        for i in range(T):
            if not np.isnan(self.sensor_data_norm[i]):
                prob0_full[i, :] = prob0[j, :]
                j += 1
            else:
                prob0_full[i, :] = 0.0

        self.sensor_data_norm1 = np.concatenate([feat, prob0_full], axis=1)

        print("[REFRESH] sensor_data_norm1 shape:", np.asarray(self.sensor_data_norm1).shape)

    def gen_test_data(self):
        self.test_points = []
        self.refresh_dataset(self.trainX)
        print("Begin to generate test_points!")

        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]
        begin_num = self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0] - start_num
        end_num = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0] - start_num

        iterval = self.roll
        for i in range(int((end_num - begin_num - self.pred_len) / iterval)):
            idx = begin_num + i * iterval
            if idx - self.seq_len < 0 or idx + self.pred_len >= len(self.data):
                continue
            seg = np.array(self.data[idx - self.seq_len: idx + self.pred_len])
            if not np.isnan(seg).any():
                self.test_points.append([self.data_time[idx]])

        self.test_dataloader()

    def test_dataloader(self):
        print("Begin to generate test_dataloader!")
        DATA, Label = [], []

        # 载入训练得到的窗口级 GMM（不要用临时的）
        gmm_path = os.path.join(self.expr_dir, "GMM.pt")
        self.gmm = torch.load(gmm_path, weights_only=False)

        for p in self.test_points:
            datetime = p[0]
            i = np.where(self.data_time == datetime)[0][0]

            if i - self.seq_len < 0 or i + self.pred_len >= len(self.data):
                continue
            if np.isnan(np.asarray(self.sensor_data_norm1[i - self.seq_len: i + self.pred_len])).any():
                continue

            data0 = np.array(self.sensor_data_norm1[i - self.seq_len: i], dtype=np.float32).reshape(self.seq_len, -1)

            label00 = np.array(self.sensor_data_norm[i: i + self.pred_len], dtype=np.float32)
            label0 = [[ff] for ff in label00]

            b = i
            e = i + self.pred_len

            label2 = [[ff] for ff in cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])]
            label3 = [[ff] for ff in sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])]

            label4 = np.array(self.data[i - 1: i + self.pred_len - 1], dtype=np.float32).reshape(-1, 1)
            label5 = np.array(self.data[i: i + self.pred_len], dtype=np.float32).reshape(-1, 1)

            label = np.concatenate((label0, label2), 1)
            label = np.concatenate((label, label3), 1)
            label = np.concatenate((label, label4), 1)
            label = np.concatenate((label, label5), 1)

            DATA.append(data0)
            Label.append(label)

        self.DATA_test = DATA
        self.Label_test = Label

        # 追加窗口级 GMM 概率（3维）到测试 DATA
        xx = np.array(self.DATA_test, np.float32)
        gmm_prob30 = self.gmm.predict_proba(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        gmm_prob3 = self._reorder_probs_by_weights(gmm_prob30, self.gmm.weights_, mode="gmm")

        prob0 = gmm_prob3[:, 0:1].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob1 = gmm_prob3[:, 1:2].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob2 = gmm_prob3[:, 2:3].repeat(self.seq_len, axis=1).reshape(len(gmm_prob3), -1, 1)
        prob = np.concatenate((prob0, prob1, prob2), axis=2)

        DATA = np.concatenate((DATA, prob), axis=2)

        print("Test DATA shape:", np.array(DATA).shape)
        print("Test Label shape:", np.array(Label).shape)

        dataset1 = TimeSeriesDataset(DATA, self.Label_test, self.config)
        self.test_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )

        test_dir = os.path.join(self.config.outf, self.config.name, "test")
        os.makedirs(test_dir, exist_ok=True)
        file_name = os.path.join(test_dir, "test_timestamps_24avg.tsv")
        pd.DataFrame(self.test_points, columns=["Test Start"]).to_csv(file_name, sep="\t")
        print("Test set saved to:", file_name)
        return self.test_data_loader
