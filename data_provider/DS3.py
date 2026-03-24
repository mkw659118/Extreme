#Author  :   mkw
#Time    :   2025/09/30 10:14:24
#Desc    :   Remove GMM logic and remove sin/cos from label

import random
import pandas as pd
from utils.utils2 import (
    diff_order_1,            # 一阶差分函数
    gen_month_tag,           # 生成月份标签
    gen_time_feature,        # 生成时间特征
    r_log_std_normalization, # 反向对数标准差归一化
    r_log_std_normalization_1, # 带参数的反向对数标准差归一化
)
import os
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_getitem_head import TimeSeriesDataset


class DS:
    """数据处理类，负责时间序列数据的预处理、特征工程、数据集构建和加载"""
    def __init__(self, config, trainX):
        self.config = config
        self.trainX = trainX

        self.mean = 0
        self.std = 0
        self.mini = 0
        self.train_mean = 0
        self.train_std = 0
        self.train_mini = 0

        self.tag = []
        self.sensor_data = []
        self.diff_data = []
        self.data = []

        self.level_mean = 0.0   # 原始序列(level)均值
        self.level_std = 1.0    # 原始序列(level)标准差
        self.d2_mean = 0.0      # 二阶差分均值（训练期）
        self.d2_std = 1.0       # 二阶差分标准差（训练期）

        self.data_time = []
        self.sensor_data_norm = []
        self.sensor_data_norm1 = []

        self.val_points = []
        self.test_points = []
        self.test_start_time = self.config.test_start
        self.test_end_time = self.config.test_end

        self.oversampling = int(config.oversampling)
        self.iterval = config.os_v

        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len

        self.lens = self.seq_len + self.pred_len + 1
        self.batch_size = config.bs
        self.thre1 = 0
        self.thre2 = 0
        self.os_h = config.os_s
        self.os_l = config.os_v

        self.val_data_loader = []
        self.train_data_loader = []
        self.test_data_loader = []
        self.month = []
        self.day = []
        self.hour = []

        self.expr_dir = os.path.join(self.config.outf, self.config.reservoir_sensor, "train")
        os.makedirs(self.expr_dir, exist_ok=True)

        # 读取数据
        self.read_dataset()
        self.roll = 8

        # 保存均值和标准差
        norm = [self.get_mean(), self.get_std()]
        np.savetxt(self.expr_dir + "/" + "Norm.txt", norm)
        norm = np.loadtxt(self.expr_dir + "/" + "Norm.txt", dtype=float, delimiter=None)
        print("norm is: ", norm)

        if self.config.mode == "train":
            self.val_dataloader()
            self.train_dataloader()
            self.refresh_dataset(trainX)
            print("[TEST] 构建测试集...")
            self.gen_test_data()

            print("样本第0列（核心归一化特征）:", self.sensor_data_norm1[:10, 0])
            print("样本第1列（level_norm）:", self.sensor_data_norm1[:10, 1])
            print("样本第2列（d2_norm）:", self.sensor_data_norm1[:10, 2])
            print("训练的mean:", self.mean)
            print("训练的std:", self.std)

    # ----------------------- 数据获取方法 -----------------------
    def get_trainX(self):
        return self.trainX

    def get_data(self):
        return self.data

    def get_diff_data(self):
        return self.diff_data

    def get_sensor_data(self):
        return self.sensor_data

    def get_sensor_data_norm(self):
        return self.sensor_data_norm

    def get_sensor_data_norm1(self):
        return self.sensor_data_norm1

    def get_val_data_loader(self):
        return self.val_data_loader

    def get_train_data_loader(self):
        return self.train_data_loader

    def get_val_points(self):
        return self.val_points

    def get_test_points(self):
        return self.test_points

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std

    def get_month(self):
        return self.month

    def get_day(self):
        return self.day

    def get_hour(self):
        return self.hour

    def get_tag(self):
        return self.tag
    
    def compute_train_diff_quantiles(self):
        """
        计算训练集真实差分的分位数，帮助 MoE 模型引导专家选择
        当前新版 label 结构:
            col 0 -> 核心归一化值
            col 1 -> 差分锚点列（前一时刻原始值）
            col 2 -> 原始值列（当前时刻原始值）
        所以真实差分 = col2 - col1
        """
        all_scores = []

        for label in self.Label:
            label = np.asarray(label, dtype=np.float32)  # [pred_len, 3]

            prev_raw = label[:, 1]
            curr_raw = label[:, 2]

            true_diff_raw = curr_raw - prev_raw  # [pred_len]

            # 样本级 score：未来窗口最大绝对差分
            score = np.max(np.abs(true_diff_raw))
            all_scores.append(score)

        all_scores = np.asarray(all_scores, dtype=np.float32)

        q50 = float(np.quantile(all_scores, 0.50))
        q80 = float(np.quantile(all_scores, 0.80))
        q95 = float(np.quantile(all_scores, 0.95))

        print(f"[Route Quantiles] q50={q50:.6f}, q80={q80:.6f}, q95={q95:.6f}")

        self.route_scores = all_scores
        self.route_q50 = q50
        self.route_q80 = q80
        self.route_q95 = q95

        return q50, q80, q95, all_scores
    
    def build_route_labels(self):
        """
        根据训练集的分位数生成 route_label
        0 -> 普通样本
        1 -> 大变化样本
        """
        route_labels = []
        for score in self.route_scores:
            # 使用 q80 划分普通样本和大变化样本
            if score <= self.route_q80:
                route_label = 0  # 普通样本
            else:
                route_label = 1  # 大变化样本
            route_labels.append(route_label)

        self.route_labels = np.asarray(route_labels, dtype=np.int64)
        return self.route_labels
    
    def build_route_labels_multi(self):
        """
        四分类 route label:
            0 -> <= q50
            1 -> (q50, q80]
            2 -> (q80, q95]
            3 -> > q95
        """
        route_labels = []
        for score in self.route_scores:
            if score <= self.route_q50:
                route_label = 0
            elif score <= self.route_q80:
                route_label = 1
            elif score <= self.route_q95:
                route_label = 2
            else:
                route_label = 3
            route_labels.append(route_label)

        self.route_labels = np.asarray(route_labels, dtype=np.int64)
        return self.route_labels

    # ----------------------- 数据读取与预处理 -----------------------
    def read_dataset(self):
        """
        从数据文件读取数据集，进行预处理，为时间序列生成标签
        """
        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]
        print("for sensor ", self.config.reservoir_sensor, "start_num is: ", start_num)

        train_end = (
            self.trainX[self.trainX["datetime"] == self.config.train_end].index.values[0] - start_num
        )
        print("train set total length is : ", train_end)

        self.sensor_data = self.trainX[start_num: train_end + start_num]

        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))

        # ================= 二阶差分特征 =================
        d2 = np.zeros_like(self.data, dtype=float)
        d2[2:] = self.data[2:] - 2.0 * self.data[1:-1] + self.data[:-2]

        self.d2_mean = np.nanmean(d2)
        self.d2_std = np.nanstd(d2)
        if (self.d2_std == 0) or np.isnan(self.d2_std):
            self.d2_std = 1.0
        d2_norm = (d2 - self.d2_mean) / self.d2_std
        d2_norm = d2_norm.reshape(-1, 1)

        # ================= 原始序列(level)标准化特征 =================
        self.level_mean = np.nanmean(self.data)
        self.level_std = np.nanstd(self.data)
        if (self.level_std == 0) or np.isnan(self.level_std):
            self.level_std = 1.0
        level_norm = (self.data - self.level_mean) / self.level_std
        level_norm = level_norm.reshape(-1, 1)

        # 一阶差分
        self.diff_data = diff_order_1(self.data)

        print("看看使用了全体数据的均值还是训练数据")
        print(len(self.data))
        print("结束》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》")

        # 核心归一化特征
        self.sensor_data_norm, self.mean, self.std, self.mini = r_log_std_normalization(self.data)
        self.sensor_data_norm1 = np.array([[ff] for ff in self.sensor_data_norm], dtype=np.float32)

        # ===== 用经验分位数替代原先 GMM 阈值，保留 oversampling 逻辑 =====
        clean_data = []
        for ii in range(len(self.sensor_data_norm)):
            if (self.sensor_data_norm[ii] is not None) and (np.isnan(self.sensor_data_norm[ii]) != 1):
                clean_data.append(self.sensor_data_norm[ii])

        clean_data = np.array(clean_data, dtype=np.float32)
        if len(clean_data) > 0:
            self.thre1 = np.percentile(clean_data, 10)
            self.thre2 = np.percentile(clean_data, 90)
        else:
            self.thre1 = 0.0
            self.thre2 = 0.0

        print("oversampling thresholds are: ", self.thre1, self.thre2)

        # 拼接 level_norm 和 d2_norm
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, level_norm), axis=1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, d2_norm), axis=1)

        # 时间相关特征（保留生成，虽然 label 不再使用 sin/cos）
        self.tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

    def val_dataloader(self):
        """
        生成验证集数据加载器
        """
        print("Begin to generate val_dataloader!")

        random.seed(self.config.val_seed)

        DATA = []
        Label = []
        ii = 0

        while ii < self.config.val_size:
            i = random.randint(self.pred_len, len(self.data) - self.lens - 1)
            a1 = 0
            a2 = -13

            if (
                (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                and (
                    self.tag[i + self.seq_len] <= a1
                    or a2 < self.tag[i + self.seq_len] < 0
                    or 2 <= self.tag[i + self.seq_len] <= 3
                )
            ):
                j = i + self.seq_len

                for k in range(1, self.seq_len + self.pred_len):
                    if j - k >= 0:
                        self.tag[j - k] = 3
                    if j + k < len(self.tag):
                        self.tag[j + k] = 3
                self.tag[j] = 2

                point = self.data_time[i + self.seq_len]
                self.val_points.append([point])

                data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                label00 = np.array(self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)])
                label0 = [[ff] for ff in label00]

                label4 = np.array(
                    self.data[(i + self.seq_len - 1): (i + self.seq_len + self.pred_len - 1)]
                ).reshape(-1, 1)
                label5 = np.array(
                    self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
                ).reshape(-1, 1)

                # label: [核心归一化值, 差分锚点列, 原始值列]
                label = np.concatenate((label0, label4), 1)
                label = np.concatenate((label, label5), 1)

                DATA.append(data0)
                Label.append(label)
                ii += 1

        self.DATA_val = DATA
        self.Label_val = Label

        print("Validation DATA shape, ", np.array(DATA).shape)
        print("Validation Label, ", np.array(Label).shape)

        dataset1 = TimeSeriesDataset(DATA, self.Label_val, self.config, route_labels=None)
        self.val_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )

        self.config.name = "%s" % (self.config.data_model)
        val_dir = os.path.join(self.config.outf, self.config.name, "val")
        os.makedirs(val_dir, exist_ok=True)
        file_name = os.path.join(val_dir, "validation_timestamps_24avg.tsv")

        pd_temp = pd.DataFrame(data=self.val_points, columns=["Hold Out Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("val set saved to : ", file_name)

    def train_dataloader(self):
        """
        生成训练集数据加载器
        """
        print("Begin to generate train_dataloader!")
        DATA = []
        Label = []

        random.seed(self.config.train_seed)
        ii = 0
        jj = 0

        while ii < self.config.train_volume:
            i = random.randint(
                self.pred_len * 4,
                len(self.sensor_data_norm) - 31 * self.pred_len * 4 - 1
            )

            pre1 = np.array(
                self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
            )
            a1 = 0
            a2 = -13

            a3 = 0
            max_index = 0

            if np.max(pre1) > self.thre2:
                a3 = self.os_h
                max_index = np.argmax(pre1)
            elif np.min(pre1) < self.thre1:
                a3 = self.os_l
                max_index = np.argmin(pre1)

            a5 = self.iterval

            # 过采样逻辑
            if (
                (jj < self.config.train_volume * (self.oversampling / 100))
                and (np.max(pre1) > self.thre2 or np.min(pre1) < self.thre1)
                and (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                and (
                    self.tag[i + self.seq_len] <= a1
                    or a2 < self.tag[i + self.seq_len] < 0
                )
            ):
                if a3 > 0:
                    i = i + max_index - 1
                    i = i - a3 * a5

                for kk in range(a3):
                    i = i + a5

                    if (i > len(self.data) - 31 * self.pred_len * 4 - 1 or i < self.pred_len * 4):
                        continue

                    if (
                        not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any()
                        and self.tag[i + self.seq_len] != 2
                        and self.tag[i + self.seq_len] != 3
                        and self.tag[i + self.seq_len] != 4
                    ):
                        Ltr = i
                        Rtr = i + self.seq_len + self.pred_len
                        win_tags = np.array(self.tag[Ltr:Rtr])
                        if ((win_tags == 2).any() or (win_tags == 3).any()):
                            continue

                        data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                        label00 = np.array(
                            self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
                        )
                        label0 = [[ff] for ff in label00]

                        label4 = np.array(
                            self.data[(i + self.seq_len - 1): (i + self.seq_len + self.pred_len - 1)]
                        ).reshape(-1, 1)
                        label5 = np.array(
                            self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
                        ).reshape(-1, 1)

                        label = np.concatenate((label0, label4), 1)
                        label = np.concatenate((label, label5), 1)

                        self.tag[i + self.seq_len] = 4
                        jj += 1
                        DATA.append(data0)
                        Label.append(label)

            # 非过采样数据
            if (
                (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                and (self.tag[i + self.seq_len] <= a1 or a2 < self.tag[i + self.seq_len] < 0)
            ):
                Ltr = i
                Rtr = i + self.seq_len + self.pred_len
                win_tags = np.array(self.tag[Ltr:Rtr])
                if ((win_tags == 2).any() or (win_tags == 3).any()):
                    continue

                data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                label00 = np.array(
                    self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
                )
                label0 = [[ff] for ff in label00]

                label4 = np.array(
                    self.data[(i + self.seq_len - 1): (i + self.seq_len + self.pred_len - 1)]
                ).reshape(-1, 1)
                label5 = np.array(
                    self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]
                ).reshape(-1, 1)

                label = np.concatenate((label0, label4), 1)
                label = np.concatenate((label, label5), 1)

                DATA.append(data0)
                Label.append(label)

                self.tag[i + self.seq_len] = 4
                ii += 1

        self.DATA = DATA
        self.Label = Label

        print("Train DATA shape, ", np.array(DATA).shape)
        print("Train Label, ", np.array(Label).shape)
        print("训练集数据的选取长度是： ", len(DATA))
        print("训练集标签的选取长度是： ", len(self.Label))

        # ===== 计算训练集分位数 =====
        self.compute_train_diff_quantiles()
        self.build_route_labels_multi()  # 生成路由标签
        print("route_labels shape:", self.route_labels.shape)
        print("route label counts:", np.bincount(self.route_labels))

        dataset1 = TimeSeriesDataset(DATA, self.Label, self.config, route_labels=self.route_labels)
        self.train_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )

    # ----------------------- 数据集刷新 -----------------------
    def refresh_dataset(self, trainX):
        """
        刷新数据集，使用已有的归一化参数(均值和标准差)
        """
        print("刷新数据集********************")
        self.trainX = trainX

        start_num = self.trainX[
            self.trainX["datetime"] == self.config.start_point
        ].index.values[0]
        print("for sensor ", self.config.reservoir_sensor, "start_num is: ", start_num)

        train_end = (
            self.trainX[self.trainX["datetime"] == self.config.train_end].index.values[0] - start_num
        )
        print("train set total length is : ", train_end)

        k = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
        self.sensor_data = self.trainX[start_num:k]
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))

        self.sensor_data_norm = r_log_std_normalization_1(self.data, self.mean, self.std)
        self.sensor_data_norm1 = np.array([[ff] for ff in self.sensor_data_norm], dtype=np.float32)

        # 二阶差分
        d2 = np.zeros_like(self.data, dtype=float)
        d2[2:] = self.data[2:] - 2.0 * self.data[1:-1] + self.data[:-2]
        d2_norm = (d2 - self.d2_mean) / self.d2_std
        d2_norm = d2_norm.reshape(-1, 1)

        # level 标准化
        level_norm = (self.data - self.level_mean) / self.level_std
        level_norm = level_norm.reshape(-1, 1)

        # 拼接特征
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, level_norm), axis=1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, d2_norm), axis=1)

        # 更新时间特征
        self.tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

    def gen_test_data(self):
        self.test_points = []
        self.refresh_dataset(self.trainX)
        print("Begin to generate test_points!")

        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]

        begin_num = (
            self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0] - start_num
        )
        end_num = (
            self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0] - start_num
        )

        iterval = self.roll

        for i in range(int((end_num - begin_num - self.pred_len) / iterval)):
            point = self.data_time[begin_num + i * iterval]
            if not np.isnan(
                np.array(
                    self.data[
                        begin_num + i * iterval - self.seq_len:
                        begin_num + i * iterval + self.pred_len
                    ]
                )
            ).any():
                self.test_points.append([point])

        self.test_dataloader()

    def test_dataloader(self):
        """
        生成测试集数据加载器
        """
        print("Begin to generate test_dataloader!")
        DATA = []
        Label = []

        for point_idx in range(len(self.test_points)):
            datetime = self.test_points[point_idx][0]
            i = np.where(self.data_time == datetime)[0][0]

            if np.isnan(self.sensor_data_norm1[i - self.seq_len: i + self.pred_len]).any():
                continue

            data0 = np.array(self.sensor_data_norm1[i - self.seq_len: i]).reshape(self.seq_len, -1)
            label00 = np.array(self.sensor_data_norm[i: i + self.pred_len])
            label0 = [[ff] for ff in label00]

            label4 = np.array(self.data[(i - 1): (i + self.pred_len - 1)]).reshape(-1, 1)
            label5 = np.array(self.data[i: i + self.pred_len]).reshape(-1, 1)

            label = np.concatenate((label0, label4), 1)
            label = np.concatenate((label, label5), 1)

            DATA.append(data0)
            Label.append(label)

        self.DATA_test = DATA
        self.Label_test = Label

        print("Test DATA shape, ", np.array(DATA).shape)
        print("Test Label, ", np.array(Label).shape)

        dataset1 = TimeSeriesDataset(DATA, self.Label_test, self.config, route_labels=None)
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

        pd_temp = pd.DataFrame(data=self.test_points, columns=["Test Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("Test set saved to : ", file_name)

        return self.test_data_loader