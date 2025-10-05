# # coding : utf-8
# # Author : yuxiang Zeng
# import collections
# from utils.utils import *
# import textwrap


# class MetricsPlotter:
#     def __init__(self, filename, config):
#         self.config = config
#         self.fileroot = f'./results/{config.model}/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '/fig/'
#         os.makedirs(self.fileroot, exist_ok=True)
#         exper_time = time.strftime('%H_%M_%S', time.localtime(time.time())) + '_'
#         self.exper_filename = self.fileroot + exper_time + filename
#         self.all_rounds_results = []
#         self.one_round_results = collections.defaultdict(list)

#     def append_epochs(self, train_loss, metrics):
#         metrics['train_loss'] = train_loss.cpu()
#         metrics = {'train_loss': metrics['train_loss'], **metrics}
#         for key, values in metrics.items():
#             self.one_round_results[key].append(values)

#     def append_round(self):
#         self.all_rounds_results.append(self.one_round_results)

#     def reset_round(self):
#         self.one_round_results = collections.defaultdict(list)

#     def record_metric(self, metrics):
#         import matplotlib.pyplot as plt
#         # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置全局字体为 'Arial Unicode MS'
#         num_rounds = len(self.all_rounds_results)
#         num_metrics = len(self.all_rounds_results[0])
#         fig, axs = plt.subplots(num_rounds, num_metrics, figsize=(num_metrics * 5, num_rounds * 5))

#         for I in range(num_rounds):
#             for j, (key, values) in enumerate(self.all_rounds_results[I].items()):
#                 ax = axs[I, j] if num_rounds > 1 else axs[j]
#                 ax.plot(range(1, len(values) + 1), values, label=key, linestyle='-', marker='o', markersize=3)
#                 ax.set_title(key)
#                 ax.set_xlabel('Epoch')
#                 ax.set_ylabel(f'Round {I + 1} Metric Value')
#                 ax.legend()

#         # 处理要显示的文本，实现自动换行
#         wrapper = textwrap.TextWrapper(width=90)  # 每行最多90个字符
#         # config_text = "\n".join(wrapper.wrap(str(self.config.__dict__)))

#         metrics_text = "\n".join(
#             [f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}' for key in metrics]
#         )

#         # combined_text = f"{config_text}\n\n{metrics_text}"
#         combined_text = f"{metrics_text}"

#         plt.rcParams['font.size'] = 24  # 设置全局字体大小为 24
#         plt.figtext(0.5, -0.2, combined_text, ha='center', va='center')
#         plt.tight_layout()
#         plt.savefig(f'{self.exper_filename}.pdf', bbox_inches='tight', pad_inches=0.5)


# coding : utf-8
# Author : yuxiang Zeng
import collections
import os
import time
import textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.utils import *  # 假设已包含必要的工具函数


class MetricsPlotter:
    def __init__(self, filename, config):
        self.config = config
        self.fileroot = f'./results/{config.model}/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '/fig/'
        os.makedirs(self.fileroot, exist_ok=True)
        exper_time = time.strftime('%H_%M_%S', time.localtime(time.time())) + '_'
        self.exper_filename = self.fileroot + exper_time + filename
        self.all_rounds_results = []
        self.one_round_results = collections.defaultdict(list)

    def append_epochs(self, train_loss, metrics):
        # 修复：只对张量执行.cpu()操作，浮点数直接使用
        if isinstance(train_loss, torch.Tensor):
            metrics['train_loss'] = train_loss.cpu()  # 张量转CPU
        else:
            metrics['train_loss'] = train_loss  # 直接使用浮点数
        
        # 合并指标
        metrics = {'train_loss': metrics['train_loss'], **metrics}
        
        # 记录指标
        for key, values in metrics.items():
            self.one_round_results[key].append(values)

    def append_round(self):
        self.all_rounds_results.append(self.one_round_results)

    def reset_round(self):
        self.one_round_results = collections.defaultdict(list)

    def record_metric(self, metrics):
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 如需显示中文可启用
        num_rounds = len(self.all_rounds_results)
        if num_rounds == 0:
            return  # 无数据时不绘图
        
        num_metrics = len(self.all_rounds_results[0])
        fig, axs = plt.subplots(num_rounds, num_metrics, figsize=(num_metrics * 5, num_rounds * 5))

        for i in range(num_rounds):
            for j, (key, values) in enumerate(self.all_rounds_results[i].items()):
                # 处理子图索引（单轮或多轮情况）
                ax = axs[i, j] if num_rounds > 1 else axs[j]
                ax.plot(range(1, len(values) + 1), values, label=key, linestyle='-', marker='o', markersize=3)
                ax.set_title(key)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'Round {i + 1} Metric Value')
                ax.legend()

        # 处理文本换行
        wrapper = textwrap.TextWrapper(width=90)
        metrics_text = "\n".join(
            [f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}' for key in metrics]
        )

        plt.rcParams['font.size'] = 12  # 调整字体大小（原24可能过大）
        plt.figtext(0.5, -0.1, metrics_text, ha='center', va='center')
        plt.tight_layout()
        plt.savefig(f'{self.exper_filename}.pdf', bbox_inches='tight', pad_inches=0.5)
        plt.close()  # 关闭画布释放资源
