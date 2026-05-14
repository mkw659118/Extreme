#Author  :   mkw
#Time    :   2025/09/30 10:14:24
#Desc    :   Remove GMM logic and remove sin/cos from label

import random
import pandas as pd
from utils.utils2 import (
    diff_order_1,            # 一阶差分函数
    gen_month_tag,           # 生成月份标签
    gen_time_feature,        # 生成时间特征
    std_normalization, # 反向对数标准差归一化
    inverse_std_normalization # 反向标准化函数
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
        self.train_mean = 0
        self.train_std = 0
        self.train_mini = 0

        self.tag = []
        self.sensor_data = []
        
        self.data = []

        self.level_mean = 0.0   # 原始序列(level)均值
        self.level_std = 1.0    # 原始序列(level)标准差
        
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
        self.roll = self.config.pred_len
        print("测试集滚动预测步长为: ", self.roll)

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
            print("训练的mean:", self.mean)
            print("训练的std:", self.std)

    # ----------------------- 数据获取方法 -----------------------
    def get_trainX(self):
        return self.trainX

    def get_data(self):
        return self.data
    
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

        # ================= 原始序列(level)标准化特征 =================
        self.level_mean = np.nanmean(self.data)
        self.level_std = np.nanstd(self.data)
        if (self.level_std == 0) or np.isnan(self.level_std):
            self.level_std = 1.0
        level_norm = (self.data - self.level_mean) / self.level_std
        level_norm = level_norm.reshape(-1, 1)

        print(len(self.data))
        print("结束》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》")

        # 核心归一化特征（单一维度）
        self.sensor_data_norm, self.mean, self.std = std_normalization(self.data)
        self.sensor_data_norm1 = np.array(self.sensor_data_norm, dtype=np.float32).reshape(-1, 1)
        print("核心归一化特征前10行:", self.sensor_data_norm1[:10, 0])

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

        # 仅使用核心归一化特征（单一维度）

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

                # label: 仅核心归一化值
                label = np.array(label0, dtype=np.float32)

                DATA.append(data0)
                Label.append(label)
                ii += 1

        self.DATA_val = DATA
        self.Label_val = Label

        print("Validation DATA shape, ", np.array(DATA).shape)
        print("Validation Label, ", np.array(Label).shape)

        dataset1 = TimeSeriesDataset(DATA, self.Label_val, self.config)
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

                        # label: 仅核心归一化值
                        label = np.array(label0, dtype=np.float32)

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

                # label: 仅核心归一化值
                label = np.array(label0, dtype=np.float32)

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

        # ===== 计算训练集点级 |diff| q90（用于 Tail 指标） =====
        all_diff_vals = []
        for label in self.Label:
            arr = np.asarray(label, dtype=np.float32)
            # 单一维度下，用相邻差分近似 |diff|
            diff_vals = np.diff(arr[:, 0], prepend=arr[0, 0])
            all_diff_vals.append(np.abs(diff_vals))
        if len(all_diff_vals) > 0:
            all_diff_vals = np.concatenate(all_diff_vals, axis=0)
            self.tail_q90 = float(np.quantile(all_diff_vals, 0.90))
        else:
            self.tail_q90 = 0.0
        setattr(self.config, 'tail_q90', self.tail_q90)
        print(f"[Tail Threshold] |diff| q90={self.tail_q90:.6f}")


        dataset1 = TimeSeriesDataset(DATA, self.Label, self.config)
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

        std = self.std if self.std != 0 else 1.0
        self.sensor_data_norm = (self.data - self.mean) / std
        self.sensor_data_norm1 = np.array(self.sensor_data_norm, dtype=np.float32).reshape(-1, 1)

        # 仅使用核心归一化特征（单一维度）

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

            # label: 仅核心归一化值
            label = np.array(label0, dtype=np.float32)

            DATA.append(data0)
            Label.append(label)

        self.DATA_test = DATA
        self.Label_test = Label

        print("Test DATA shape, ", np.array(DATA).shape)
        print("Test Label, ", np.array(Label).shape)

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

        pd_temp = pd.DataFrame(data=self.test_points, columns=["Test Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("Test set saved to : ", file_name)

        return self.test_data_loader