#Author  :   mkw 
#Time    :   2025/09/30 10:14:24
#Desc    :   None

import random
import pandas as pd
from utils.utils2 import (
    diff_order_1,           # 一阶差分函数
    gen_month_tag,          # 生成月份标签
    gen_time_feature,       # 生成时间特征
    cos_date,               # 生成日期余弦特征
    sin_date,               # 生成日期正弦特征   
    r_log_std_normalization, # 反向对数标准差归一化
    r_log_std_normalization_1, # 带参数的反向对数标准差归一化
)
import os
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture  # 高斯混合模型，用于概率分布建模
import numpy as np
from data_provider.data_getitem import TimeSeriesDataset

class DS:
    """数据处理类，负责时间序列数据的预处理、特征工程、数据集构建和加载"""
    def __init__(self, config, trainX):
        self.config = config
        self.trainX = trainX
        self.mean = 0          # 数据均值
        self.std = 0           # 数据标准差
        self.mini = 0          # 数据最小值
        self.train_mean = 0          # 数据均值
        self.train_std = 0           # 数据标准差
        self.train_mini = 0          # 数据最小值
        self.tag = []          # 时间序列标签
        self.sensor_data = []  # 传感器原始数据
        self.diff_data = []    # 差分后数据
        self.data = []         # 数值数据
        self.level_mean = 0.0   # NEW: 原始序列(level)均值
        self.level_std  = 1.0   # NEW: 原始序列(level)标准差
        self.d2_mean = 0.0   # NEW: 二阶差分均值（训练期）
        self.d2_std  = 1.0   # NEW: 二阶差分标准差（训练期）


        
        self.data_time = []    # 时间戳数据
        self.sensor_data_norm = []    # 归一化后数据
        self.sensor_data_norm1 = []   # 扩展特征后的归一化数据

        self.val_points = []   # 验证集时间点
        self.test_points = []  # 测试集时间点
        self.test_start_time = self.config.test_start  # 测试开始时间
        self.test_end_time = self.config.test_end      # 测试结束时间
        self.gm3 = GaussianMixture(n_components=3)  # 三成分高斯混合模型，用于异常检测

        self.oversampling = int(config.oversampling)  # 过采样率
        # self.oversampling = 0  # 过采样率
        self.iterval = config.os_v     # 过采样间隔

        self.seq_len = self.config.seq_len      # 输入序列长度(天数)
        self.pred_len = self.config.pred_len   # 预测序列长度(天数)

        self.lens = self.seq_len + self.pred_len + 1  # 总序列长度
        self.batch_size = config.bs          # 批量大小
        self.thre1 = 0                             # 阈值1
        self.thre2 = 0                             # 阈值2
        self.os_h = config.os_s                    # 过采样上限
        self.os_l = config.os_v                    # 过采样下限
        self.gmm_l = self.pred_len                 # GMM模型长度
        
        self.val_data_loader = []                  # 验证集数据加载器
        self.train_data_loader = []                # 训练集数据加载器
        self.test_data_loader = []                 # 训练集数据加载器
        self.month = []                            # 月份特征
        self.day = []                              # 日期特征
        self.hour = []                             # 小时特征

        self.expr_dir = os.path.join(self.config.outf, self.config.reservoir_sensor, "train")  # 实验目录
        os.makedirs(self.expr_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        
        # 这里是数据
        self.read_dataset()                        # 读取并预处理数据集
        self.roll = 8                              # 滚动间隔

        # 保存数据集的均值和标准差
        norm = []
        norm.append(self.get_mean())
        norm.append(self.get_std())
        np.savetxt(self.expr_dir + "/" + "Norm.txt", norm)
        norm = np.loadtxt(self.expr_dir + "/" + "Norm.txt", dtype=float, delimiter=None)
        print("norm is: ", norm)
        

        if self.config.mode == "train":
            self.train_temp_gmm()         # 临时训练GMM
            self.val_dataloader()         # 生成验证集数据加载器
            self.train_dataloader()       # 生成训练集数据加载器
            self.refresh_dataset(trainX)  # 刷新数据集
            print("[TEST] 构建测试集...")
            self.gen_test_data()          # 构建 test_dataloader
            print("样本第0列：", self.sensor_data_norm1[:10, 0])  # 异常概率
            print("样本第1列：", self.sensor_data_norm1[:10, 1])  # 核心数值特征
            print("训练的mean:",self.mean)
            print("训练的std:",self.std)

    # ----------------------- 数据获取方法 -----------------------
    def get_trainX(self):
        """获取原始训练数据"""
        return self.trainX

    def get_data(self):
        """获取数值数据"""
        return self.data

    def get_diff_data(self):
        """获取差分后数据"""
        return self.diff_data

    def get_sensor_data(self):
        """获取传感器原始数据"""
        return self.sensor_data

    def get_sensor_data_norm(self):
        """获取归一化后数据"""
        return self.sensor_data_norm

    def get_sensor_data_norm1(self):
        """获取扩展特征后的归一化数据"""
        return self.sensor_data_norm1

    def get_val_data_loader(self):
        """获取验证集数据加载器"""
        return self.val_data_loader

    def get_train_data_loader(self):
        """获取训练集数据加载器"""
        return self.train_data_loader

    def get_val_points(self):
        """获取验证集时间点"""
        return self.val_points

    def get_test_points(self):
        """获取测试集时间点"""
        return self.test_points

    def get_mean(self):
        """获取数据均值"""
        return self.mean

    def get_std(self):
        """获取数据标准差"""
        return self.std

    def get_month(self):
        """获取月份特征"""
        return self.month

    def get_day(self):
        """获取日期特征"""
        return self.day

    def get_hour(self):
        """获取小时特征"""
        return self.hour

    def get_tag(self):
        """获取时间序列标签"""
        return self.tag

    # ----------------------- 数据读取与预处理 -----------------------
    def read_dataset(self):
        """
        从数据文件读取数据集，进行预处理，为时间序列生成标签(0表示无值，1表示有效值)
        """
        # 找到起始时间点的索引
        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]
        print("for sensor ", self.config.reservoir_sensor, "start_num is: ", start_num)
        
        # 找到训练结束时间点的索引
        train_end = (self.trainX[self.trainX["datetime"] == self.config.train_end].index.values[0] - start_num)
        print("train set total length is : ", train_end)

        # 加载整个数据集
        self.sensor_data = self.trainX[start_num: train_end + start_num]
        
        # 缺失值?
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        
        # ================= NEW: 二阶差分(基于原始序列 level) =================
        # d2[t] = x[t] - 2*x[t-1] + x[t-2] ，前两个位置无定义，这里用 0 补齐（不引入未来信息）
        d2 = np.zeros_like(self.data, dtype=float)
        d2[2:] = self.data[2:] - 2.0 * self.data[1:-1] + self.data[:-2]

        # 用训练段统计量标准化（忽略 NaN）
        self.d2_mean = np.nanmean(d2)
        self.d2_std  = np.nanstd(d2)
        if (self.d2_std == 0) or np.isnan(self.d2_std):
            self.d2_std = 1.0
        d2_norm = (d2 - self.d2_mean) / self.d2_std
        d2_norm = d2_norm.reshape(-1, 1)
        # =====================================================================

        
        # ================= NEW: 计算原始序列(level)的标准化特征 =================
        self.level_mean = np.nanmean(self.data)
        self.level_std  = np.nanstd(self.data)
        if (self.level_std == 0) or np.isnan(self.level_std):
            self.level_std = 1.0
        level_norm = (self.data - self.level_mean) / self.level_std   # 与 self.data 同长度，NaN 会保留
        level_norm = level_norm.reshape(-1, 1)
        # =====================================================================

        
        # 
        self.diff_data = diff_order_1(self.data)  # 计算一阶差分
        print("看看使用了全体数据的均值还是训练数据")
        print(len(self.data))
        print("结束》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》")
        # 对数据进行反向对数标准差归一化，并保存归一化参数
        self.sensor_data_norm, self.mean, self.std, self.mini = r_log_std_normalization(self.data)
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]

        gmm_input = self.sensor_data_norm

        # 清理数据，去除NaN值
        clean_data = []
        for ii in range(len(self.sensor_data_norm)):
            if (self.sensor_data_norm[ii] is not None) and (np.isnan(self.sensor_data_norm[ii]) != 1):
                clean_data.append(self.sensor_data_norm[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
        
        
        # # 训练数据集级别的三成分高斯混合模型，用于异常检测
        self.gm3.fit(sensor_data_prob)
        torch.save(self.gm3, self.expr_dir + "/" + "GM3.pt")
        
        # self.gm_means = np.squeeze(self.gm3.means_)
        # self.z0 = np.min(self.gm_means)
        # self.z1 = np.median(self.gm_means)
        # self.z2 = np.max(self.gm_means)

        # # 计算异常检测阈值
        # self.thre1 = (self.z0 + self.z1) / 2
        # self.thre2 = (self.z1 + self.z2) / 2
        
        # print("gm3.means are: ", self.gm_means)
        # print("z : ", self.z0, self.z1, self.z2)
        
        clean_data = np.array(clean_data, dtype=np.float32)
        if len(clean_data) > 0:
            self.thre1 = np.percentile(clean_data, 10)
            self.thre2 = np.percentile(clean_data, 90)
        else:
            self.thre1 = 0.0
            self.thre2 = 0.0
        print("thre1 is: ", self.thre1)
        print("thre2 is: ", self.thre2) 
        
        print("gm3.covariances are: ", self.gm3.covariances_)
        print("gm3.weights are: ", self.gm3.weights_)
        weights3 = self.gm3.weights_
        
        
        # 计算数据点属于分布的概率和异常概率
        data_prob3 = self.gm3.predict_proba(sensor_data_prob)
        prob_in_distribution3 = (data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2])
        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))

        # 恢复异常概率数组，保持与原始数据相同的长度
        recover_data = []
        temp = 0
        jj = 0
        for ii in range(len(self.sensor_data_norm)):
            if (self.sensor_data_norm[ii] is not None) and (np.isnan(self.sensor_data_norm[ii]) != 1):
                recover_data.append(prob_like_outlier3[jj])
                jj = jj + 1
            else:
                recover_data.append(self.sensor_data_norm[ii])
        prob_like_outlier3 = np.array(recover_data, np.float32).reshape(len(self.sensor_data_norm), 1)
        
        # 将异常概率作为新特征添加到归一化数据中
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, prob_like_outlier3), 1)

        # 训练另一个高斯混合模型，使用随机采样数据
        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
        
        self.gmm0 = GaussianMixture(n_components=3)
        series = []
        random.seed(self.config.val_seed)
        for ggg in range(200000):
            g0 = random.randint(0, len(gmm_input) - self.gmm_l)
            if not np.isnan(gmm_input[g0]).any():
                series.append([gmm_input[g0]])
        self.gmm0.fit(np.array(series).reshape(-1, 1)) 
        torch.save(self.gmm0, self.expr_dir + "/" + "GMM0.pt")
        self.gmm0_means = np.squeeze(self.gmm0.means_)
        print("gmm0.means are: ", self.gmm0_means)
        print("gmm0.weights are: ", self.gmm0.weights_)
        weights3 = self.gmm0.weights_
        
        # 预测每个数据点在各成分上的后验概率，并按权重排序
        data_prob30 = self.gmm0.predict_proba(sensor_data_prob) 
        order1 = np.argmax(weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)
        
        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)

        # 恢复概率数组，保持与原始数据相同的长度
        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32)
        
        # 将排序后的后验概率作为新特征添加到归一化数据中
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 0:1]), 1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 1:2]), 1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 2:3]), 1)
        print("sensor_data_norm1, ", self.sensor_data_norm1)
        print("Finish prob indicator generating.")
        
        # NEW: 将原始序列(level)作为最后一维特征拼接进输入
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, level_norm), axis=1)
        # NEW: 将二阶差分特征追加到输入末尾（不改变原有列顺序）
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, d2_norm), axis=1)



        # 生成时间相关特征
        self.tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        # 生成日期的正弦和余弦特征，用于周期性表示
        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]
    
    def train_temp_gmm(self):
        """临时训练GMM，供验证集生成时使用，增强版：解决样本不足问题"""
        # 1. 提取原始数据并清洗NaN值
        temp_data = np.array(self.sensor_data_norm1)
        # 过滤掉包含NaN的行（确保每一行都是有效数据）
        clean_temp_data = temp_data[~np.isnan(temp_data).any(axis=1)]
        window_size = self.gmm_l  # 窗口大小=预测长度（如8）
        min_required_samples = 2  # GMM训练至少需要2个样本

        # 2. 检查清洗后的数据量是否足够
        if len(clean_temp_data) < window_size:
            # 情况1：连一个完整窗口的长度都不够
            raise ValueError(
                f"清洗后的数据量不足！需要至少{window_size}个有效时间步，"
                f"但仅找到{len(clean_temp_data)}个。请检查数据质量或减小pred_len。"
            )

        # 3. 用滑动窗口生成样本（核心修复）
        # 计算最大可能的样本数（滑动窗口步数）
        max_possible_samples = len(clean_temp_data) - window_size + 1
        # 实际取的样本数：取最大可能样本数和1000的较小值（避免样本过多导致计算慢）
        n_samples = min(max_possible_samples, 1000)

        if n_samples < min_required_samples:
            # 情况2：样本数不足2个，尝试复制样本应急（仅作为临时方案）
            print(f"警告：有效样本数不足（{n_samples}个），将复制样本以满足GMM训练要求")
            # 先按现有样本生成数据
            temp_gmm_input = []
            for i in range(n_samples):
                window = clean_temp_data[i:i+window_size, 1:2].flatten()
                temp_gmm_input.append(window)
            # 复制样本直到满足2个
            while len(temp_gmm_input) < min_required_samples:
                temp_gmm_input.append(temp_gmm_input[-1])  # 复制最后一个样本
            temp_gmm_input = np.array(temp_gmm_input)
        else:
            # 情况3：样本数足够，正常生成
            temp_gmm_input = []
            for i in range(n_samples):
                # 取每个窗口的第1列特征（与正式训练逻辑一致），展平为1维数组
                window = clean_temp_data[i:i+window_size, 1:2].flatten()
                temp_gmm_input.append(window)
            temp_gmm_input = np.array(temp_gmm_input)

        # 4. 训练临时GMM并保存
        self.gmm = GaussianMixture(n_components=3)
        self.gmm.fit(temp_gmm_input)  # 此时样本数至少为2，满足GMM要求
        torch.save(self.gmm, self.expr_dir + "/" + "GMM.pt")
        print(f"临时GMM训练完成，使用了{len(temp_gmm_input)}个样本（窗口大小{window_size}）")
    
    def val_dataloader(self):
        """
        生成验证集数据加载器
        随机选择时间序列中的点，若为有效起始时间(序列中无NaN值，且在指定月份范围内)，标记为3
        邻近点标记为4，并将数据封装为与训练集一致的DataLoader
        """
        print("Begin to generate val_dataloader!")

        near_len = self.pred_len
        random.seed(self.config.val_seed)
        
        DATA = []
        Label = []
        ii = 0
        
        while ii < self.config.val_size:
            # 随机选择起始索引
            i = random.randint(self.pred_len, len(self.data) - self.lens - 1)
            a1 = 0
            a2 = -13
            # 检查条件：序列无NaN值，且时间标签在指定范围内
            if (
                (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                and (
                    self.tag[i + self.seq_len] <= a1
                    or a2 < self.tag[i + self.seq_len] < 0
                    or 2 <= self.tag[i + self.seq_len] <= 3
                )
            ):
                j = i + self.seq_len
                # 先标邻居，再标中心，避免中心被覆盖
                for k in range(1, self.seq_len + self.pred_len):  # 覆盖到左右各 seq_len+pred_len-1
                    if j - k >= 0:
                        self.tag[j - k] = 3
                    if j + k < len(self.tag):
                        self.tag[j + k] = 3
                self.tag[j] = 2  # 最后再标中心，避免被 3 覆盖

                point = self.data_time[i + self.seq_len]
                self.val_points.append([point])
                
                # 准备验证数据和标签（与训练集一致的格式）
                data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                label00 = np.array(self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)])
                label0 = [[ff] for ff in label00]

                b = i + self.seq_len
                e = i + self.seq_len + self.pred_len

                # 生成时间相关特征作为标签的一部分
                label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                label2 = [[ff] for ff in label2]

                label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                label3 = [[ff] for ff in label3]
                
                label4 = np.array(self.data[(i + self.seq_len - 1):(i + self.seq_len + self.pred_len - 1)]).reshape(-1, 1)
                label5 = np.array(self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]).reshape(-1, 1)

                # 合并标签的各个部分
                label = np.concatenate((label0, label2), 1)
                label = np.concatenate((label, label3), 1)
                label = np.concatenate((label, label4), 1)
                label = np.concatenate((label, label5), 1)

                DATA.append(data0)
                Label.append(label)
                ii = ii + 1

        self.DATA_val = DATA
        self.Label_val = Label
        
        # 加载预训练的样本级高斯混合模型生成概率特征
        self.gmm = torch.load(self.expr_dir + "/" + "GMM.pt", weights_only=False)
        xx = np.array(self.DATA_val, np.float32)
        gmm_prob30 = self.gmm.predict_proba(
            np.squeeze(xx[:, -1 * self.gmm_l:, 1:2])
        )
        
        # 对概率进行排序
        order1 = np.argmin(self.gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("val gmm order is, ", order1, order2, order3)
        d2 = gmm_prob30[:, order3].reshape(-1, 1)
        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)
        # 扩展概率维度以匹配训练数据的时间维度
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)
        prob = np.concatenate((prob0, prob1), 2)
        prob = np.concatenate((prob, prob2), 2)
        
        # 将新生成的概率特征添加到验证数据中
        DATA = np.concatenate((DATA, prob), 2)
        print("Validation DATA shape, ", np.array(DATA).shape)
        print("Validation Label, ", np.array(Label).shape)

        # 创建数据集和数据加载器
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
        # self.config.name = "%s" % (self.config.data_model)
        self.config.name = "%s" % (self.config.reservoir_sensor)
        val_dir = os.path.join(self.config.outf, self.config.name, "val")
        os.makedirs(val_dir, exist_ok=True)
        file_name = os.path.join(val_dir, "validation_timestamps_24avg.tsv")

        pd_temp = pd.DataFrame(data=self.val_points, columns=["Hold Out Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("val set saved to : ", file_name)



    def train_dataloader(self):
        """
        生成训练集数据加载器
        只能在val_dataloader之后运行
        随机选择时间序列中的点，若为有效起始时间(序列中无NaN值，在指定月份范围内，且标签不是3和4)，
        选择作为训练点，标记为5
        """
        print("Begin to generate train_dataloader!")
        DATA = []  # 存储训练数据
        Label = []  # 存储训练标签

        # 随机选择训练数据
        random.seed(self.config.train_seed)  # 设置随机种子保证结果可复现
        ii = 0  # 普通样本计数器
        jj = 0  # 过采样样本计数器
        
        # 循环直到收集到足够的训练样本
        while ii < self.config.train_volume:
            # 随机选择起始索引，确保有足够的上下文和预测空间
            i = random.randint(self.pred_len * 4, len(self.sensor_data_norm) - 31 * self.pred_len * 4 - 1)
            # 提取预测时间段的数据
            pre1 = np.array(
                self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)])
            a1 = 0
            a2 = -13
            
            # 判断数据是否为极端值，确定过采样参数
            if np.max(pre1) > self.thre2:
                a3 = self.os_h  # 过采样上限
                max_index = np.argmax(pre1)  # 最大值索引
            elif np.min(pre1) < self.thre1:
                a3 = self.os_l  # 过采样下限
                max_index = np.argmin(pre1)  # 最小值索引
            a5 = self.iterval  # 过采样间隔
            
            # 过采样逻辑：对极端值进行过采样以平衡数据集
            if (
                (jj < self.config.train_volume * (self.oversampling / 100))  # 过采样数量限制
                and (np.max(pre1) > self.thre2 or np.min(pre1) < self.thre1)  # 极端值判断
                and (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())  # 数据有效性检查
                and (
                    self.tag[i + self.seq_len] <= a1
                    or a2 < self.tag[i + self.seq_len] < 0
                )
            ):
                if a3 > 0:
                    # 调整索引以定位到极端值附近
                    i = i + max_index - 1
                    i = i - a3 * a5
                # 按照过采样参数生成多个样本
                for kk in range(a3):  
                    i = i + a5  # 按间隔移动索引
                    # 检查索引有效性
                    if (i > len(self.data) - 31 * self.pred_len * 4 - 1 or i < self.pred_len * 4):
                        continue
                    # 确保数据有效且未被标记为验证集或邻近区域
                    if (
                        not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any()
                        and self.tag[i + self.seq_len] != 2
                        and self.tag[i + self.seq_len] != 3
                        and self.tag[i + self.seq_len] != 4
                    ):
                        # NEW: 训练窗口与验证“禁区”是否相交？
                        Ltr = i
                        Rtr = i + self.seq_len + self.pred_len   # 等价 i + self.lens - 1
                        win_tags = np.array(self.tag[Ltr:Rtr])   # 若 self.tag 是 list，转成 np.array
                        if ((win_tags == 2).any() or (win_tags == 3).any()):
                            continue  # 命中验证中心(2)或邻域(3)，丢弃这个候选


                        # 准备训练数据和标签
                        data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                        label00 = np.array(self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)])
                        label0 = [[ff] for ff in label00]

                        b = i + self.seq_len
                        e = i + self.seq_len + self.pred_len

                        # 生成时间相关特征作为标签的一部分（周期性特征）
                        label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                        label2 = [[ff] for ff in label2]

                        label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                        label3 = [[ff] for ff in label3]

                        # 添加历史数据作为标签的一部分
                        label4 = np.array(self.data[(i+self.seq_len-1):(i + self.seq_len + self.pred_len - 1)]).reshape(-1, 1)
                        label5 = np.array(self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]).reshape(-1, 1)

                        # 合并标签的各个部分
                        label = np.concatenate((label0, label2), 1)
                        label = np.concatenate((label, label3), 1)
                        label = np.concatenate((label, label4), 1)
                        label = np.concatenate((label, label5), 1)

                        self.tag[i + self.seq_len] = 4  # 标记为已使用的训练点
                        jj = jj + 1  # 过采样计数器加1
                        DATA.append(data0)
                        Label.append(label)

            # 非过采样数据处理
            if (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any()) and (self.tag[i + self.seq_len] <= a1 or a2 < self.tag[i + self.seq_len] < 0):
                Ltr = i
                Rtr = i + self.seq_len + self.pred_len   # 等价 i + self.lens - 1
                win_tags = np.array(self.tag[Ltr:Rtr])
                if ((win_tags == 2).any() or (win_tags == 3).any()):
                    continue

                # 准备训练数据和标签（与过采样情况类似）
                data0 = np.array(self.sensor_data_norm1[i: (i + self.seq_len)]).reshape(self.seq_len, -1)
                label00 = np.array(self.sensor_data_norm[(i + self.seq_len): (i + self.seq_len + self.pred_len)])
                label0 = [[ff] for ff in label00]

                b = i + self.seq_len
                e = i + self.seq_len + self.pred_len

                # 生成时间相关特征
                label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                label2 = [[ff] for ff in label2]

                label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])
                label3 = [[ff] for ff in label3]
                
                # 添加历史数据特征
                label4 = np.array(self.data[(i + self.seq_len - 1):(i + self.seq_len + self.pred_len - 1)]).reshape(-1, 1)
                label5 = np.array(self.data[(i + self.seq_len): (i + self.seq_len + self.pred_len)]).reshape(-1, 1)

                # 合并标签
                label = np.concatenate((label0, label2), 1)
                label = np.concatenate((label, label3), 1)
                label = np.concatenate((label, label4), 1)
                label = np.concatenate((label, label5), 1)

                DATA.append(data0)
                Label.append(label)

                self.tag[i + self.seq_len] = 4  # 标记为已使用的训练点
                ii = ii + 1  # 普通样本计数器加1

        self.DATA = DATA
        self.Label = Label

        # 训练样本级高斯混合模型，生成新的概率特征
        self.gmm = GaussianMixture(n_components=3)
        xx = np.array(self.DATA, np.float32)
        # 使用训练数据的最后self.gmm_l个时间步的特征训练GMM
        self.gmm.fit(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        torch.save(self.gmm, self.expr_dir + "/" + "GMM.pt")  # 保存GMM模型
        self.gmm_means = np.squeeze(self.gmm.means_)
        print("time series gmm.weights are: ", self.gmm.weights_)
        
        # 使用GMM预测训练数据的概率分布
        gmm_prob30 = self.gmm.predict_proba(
            np.squeeze(np.array(self.DATA)[:, -1 * self.gmm_l:, 1:2])
        )
        # 对概率进行排序（按权重大小）
        order1 = np.argmin(self.gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)
        d2 = gmm_prob30[:, order3].reshape(-1, 1)
        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)
        
        # 扩展概率维度以匹配训练数据的时间维度
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)
        prob = np.concatenate((prob0, prob1), 2)
        prob = np.concatenate((prob, prob2), 2)
        
        # 将新生成的概率特征添加到训练数据中
        DATA = np.concatenate((DATA, prob), 2)
        print("Train DATA shape, ", np.array(DATA).shape)
        print("Train Label, ", np.array(Label).shape)
        print("训练集数据的选取长度是： ", len(DATA))
        print("训练集标签的选取长度是： ", len(self.Label))
        
        # ===== 计算训练集点级 |diff| q90（用于 Tail 指标） =====
        all_diff_vals = []
        for label in self.Label:
            arr = np.asarray(label, dtype=np.float32)
            # label[:, 1] = previous raw anchor, label[:, 2] = current raw value
            true_diff_raw = arr[:, 4] - arr[:, 3]
            all_diff_vals.append(np.abs(true_diff_raw))
        if len(all_diff_vals) > 0:
            all_diff_vals = np.concatenate(all_diff_vals, axis=0)
            self.tail_q90 = float(np.quantile(all_diff_vals, 0.90))
        else:
            self.tail_q90 = 0.0
        setattr(self.config, 'tail_q90', self.tail_q90)
        print(f"[Tail Threshold] |diff| q90={self.tail_q90:.6f}")



        # 创建数据集和数据加载器
        dataset1 = TimeSeriesDataset(DATA, self.Label, self.config )
        self.train_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=True,  
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,  # 使用自定义的collate函数处理数据
        )

    # ----------------------- 数据集刷新 -----------------------
    def refresh_dataset(self, trainX):
        """
        刷新数据集，使用已有的归一化参数(均值和标准差)
        :param trainX: 新的训练数据集
        """
        print("刷新数据集********************")
        self.trainX = trainX
        # 找到起始时间点的索引
        start_num = self.trainX[
            self.trainX["datetime"] == self.config.start_point
        ].index.values[0]
        print("for sensor ", self.config.reservoir_sensor, "start_num is: ", start_num)
        # 找到训练结束时间点的索引
        train_end = (self.trainX[self.trainX["datetime"] == self.config.train_end].index.values[0] - start_num)
        print("train set total length is : ", train_end)

        # 找到测试结束时间点的索引并加载数据集
        k = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
        self.sensor_data = self.trainX[start_num:k]
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        # 使用已有的均值和标准差进行归一化
        self.sensor_data_norm = r_log_std_normalization_1(self.data, self.mean, self.std)
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]
        
        # ================= NEW: 二阶差分(基于原始序列 level)，使用训练期统计量标准化 =================
        d2 = np.zeros_like(self.data, dtype=float)
        d2[2:] = self.data[2:] - 2.0 * self.data[1:-1] + self.data[:-2]
        d2_norm = (d2 - self.d2_mean) / self.d2_std
        d2_norm = d2_norm.reshape(-1, 1)
        # ==================================================================================================

        
        # ================= NEW: 用训练期的 level_mean/level_std 标准化原始序列(level) =================
        level_norm = (self.data - self.level_mean) / self.level_std
        level_norm = level_norm.reshape(-1, 1)
# =================================================================================================


        gmm_input = self.sensor_data_norm

        # 清理数据，去除NaN值
        clean_data = []
        for ii in range(len(self.sensor_data_norm)):
            if (self.sensor_data_norm[ii] is not None) and (np.isnan(self.sensor_data_norm[ii]) != 1):
                clean_data.append(self.sensor_data_norm[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
        # 使用预训练的高斯混合模型预测概率
        data_prob3 = self.gm3.predict_proba(sensor_data_prob)
        weights3 = self.gm3.weights_
        
        # 计算数据点属于分布的概率和异常概率
        prob_in_distribution3 = (data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2])
        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))

        # 恢复异常概率数组，保持与原始数据相同的长度
        recover_data = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(self.sensor_data_norm)):
            if (self.sensor_data_norm[ii] is not None) and (
                np.isnan(self.sensor_data_norm[ii]) != 1
            ):
                recover_data.append(prob_like_outlier3[jj])
                jj = jj + 1
            else:
                recover_data.append(self.sensor_data_norm[ii])
        prob_like_outlier3 = np.array(recover_data, np.float32).reshape(len(self.sensor_data_norm), 1)
        # 添加异常概率作为新特征
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, prob_like_outlier3), 1)

        # 生成点级概率特征
        clean_data = []
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                clean_data.append(gmm_input[ii])
        sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
        
        # 使用预训练的高斯混合模型预测概率并排序
        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        data_prob30 = self.gmm0.predict_proba(sensor_data_prob)
        order1 = np.argmax(weights3)
        d0 = data_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmin(weights3)
        d1 = data_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)
        data_prob3 = np.concatenate((d0, d1), 1)
        data_prob3 = np.concatenate((data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1)

        # 恢复概率数组，保持与原始数据相同的长度
        recover_prob = []
        temp = np.zeros(np.array(data_prob3[0]).shape)
        jj = 0
        for ii in range(len(gmm_input)):
            if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                recover_prob.append(data_prob3[jj])
                jj = jj + 1
            else:
                recover_prob.append(temp)
        recover_prob = np.array(recover_prob, np.float32).reshape(len(gmm_input), -1)
        # 添加排序后的概率作为新特征
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 0:1]), 1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 1:2]), 1)
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, recover_prob[:, 2:3]), 1)
        print("Finish prob indicator updating.")
        
        # NEW: 追加原始序列(level)特征到最后一列（不改变原有列顺序）
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, level_norm), axis=1)
        
        # NEW: 将二阶差分特征追加到输入末尾
        self.sensor_data_norm1 = np.concatenate((self.sensor_data_norm1, d2_norm), axis=1)


        # 更新时间相关特征
        self.tag = gen_month_tag(self.sensor_data)
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        # 生成日期的正弦和余弦特征
        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]

    def gen_test_data(self):

        self.test_points = []
        self.refresh_dataset(self.trainX)
        print("Begin to generate test_points!")

        start_num = self.trainX[self.trainX["datetime"] == self.config.start_point].index.values[0]

        begin_num = (self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0]- start_num)
        end_num = (self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0] - start_num)

        iterval = self.roll

        for i in range(int((end_num - begin_num - self.pred_len) / iterval)):  # do inference every 24 hours
            point = self.data_time[begin_num + i * iterval]
            if not np.isnan(
                np.array(
                    self.data[
                        begin_num
                        + i * iterval
                        - self.seq_len: begin_num
                        + i * iterval
                        + self.pred_len
                    ]
                )
            ).any():
                self.test_points.append([point])
        self.test_dataloader()

    def test_dataloader(self):
        """
        生成测试集数据加载器
        基于已生成的测试点，准备测试数据并封装为DataLoader
        """
        print("Begin to generate test_dataloader!")
        DATA = []
        Label = []
        
        # 加载预训练的高斯混合模型
        self.gm3 = torch.load(self.expr_dir + "/" + "GM3.pt", weights_only=False)
        self.gmm0 = torch.load(self.expr_dir + "/" + "GMM0.pt", weights_only=False)
        self.gmm = torch.load(self.expr_dir + "/" + "GMM.pt", weights_only=False)
        
        # 遍历每个测试点
        for point_idx in range(len(self.test_points)):
            # 找到测试点在数据中的索引
            datetime = self.test_points[point_idx][0]
            i = np.where(self.data_time == datetime)[0][0]
            
            # 确保数据范围内无NaN值
            if np.isnan(self.sensor_data_norm1[i-self.seq_len: i+self.pred_len]).any():
                continue
                
            # 准备测试数据和标签（格式与训练/验证集一致）
            data0 = np.array(self.sensor_data_norm1[i-self.seq_len: i]).reshape(self.seq_len, -1)
            label00 = np.array(self.sensor_data_norm[i: i+self.pred_len])
            label0 = [[ff] for ff in label00]
            
            b = i
            e = i + self.pred_len
            
            # 生成时间相关特征作为标签的一部分
            label2 = cos_date(self.month[b:e], self.day[b:e], self.hour[b:e])
            label2 = [[ff] for ff in label2]
            label3 = sin_date(self.month[b:e], self.day[b:e], self.hour[b:e])
            label3 = [[ff] for ff in label3]
            
            label4 = np.array(self.data[(i-1):(i+self.pred_len-1)]).reshape(-1, 1)
            label5 = np.array(self.data[i: i+self.pred_len]).reshape(-1, 1)
            
            # 合并标签的各个部分
            label = np.concatenate((label0, label2), 1)
            label = np.concatenate((label, label3), 1)
            label = np.concatenate((label, label4), 1)
            label = np.concatenate((label, label5), 1)
            
            DATA.append(data0)
            Label.append(label)
        
        self.DATA_test = DATA
        self.Label_test = Label
        
        # 生成概率特征（与训练/验证集一致的处理方式）
        xx = np.array(self.DATA_test, np.float32)
        gmm_prob30 = self.gmm.predict_proba(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        
        # 对概率进行排序
        order1 = np.argmin(self.gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("test gmm order is, ", order1, order2, order3)
        d2 = gmm_prob30[:, order3].reshape(-1, 1)
        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)
        
        # 扩展概率维度以匹配时间维度
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(self.seq_len, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)
        prob = np.concatenate((prob0, prob1), axis=2)
        prob = np.concatenate((prob, prob2), axis=2)
        
        # 将概率特征添加到测试数据中
        DATA = np.concatenate((DATA, prob), 2)
        
        print("Test DATA shape, ", np.array(DATA).shape)
        print("Test Label, ", np.array(Label).shape)
        
        # 创建测试数据集和数据加载器
        from data_provider.data_getitem import TimeSeriesDataset
        dataset1 = TimeSeriesDataset(DATA, self.Label_test, self.config)
        self.test_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=False,  
            num_workers=2,
            pin_memory=True,
            collate_fn=dataset1.custom_collate_fn,
        )
        
        # 保存测试集时间戳
        test_dir = os.path.join(self.config.outf, self.config.name, "test")
        os.makedirs(test_dir, exist_ok=True)
        file_name = os.path.join(test_dir, "test_timestamps_24avg.tsv")
        
        pd_temp = pd.DataFrame(data=self.test_points, columns=["Test Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("Test set saved to : ", file_name)
        return self.test_data_loader
