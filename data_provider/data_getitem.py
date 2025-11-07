#Author  :   mkw 
#Time    :   2025/09/30 14:23:08
#Desc    :   None

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, config):
        self.config = config
        self.x = x
        self.y = y
        self.id_offset = int(getattr(config, "id_offset", 0))

    def __len__(self):
        return len(self.x)
       
    def __getitem__(self, idx): 
        if self.config.model == 'patch_extreme_memory_transformer':
            x = self.x[idx]
            x_mark = x[:, 2:]
            y = self.y[idx]
            # 样本级记忆库所需的稳定ID
            sample_id = np.int64(self.id_offset + idx)

            return (
                x.astype(np.float32),
                x_mark.astype(np.float32),
                y.astype(np.float32),
                sample_id,
            )
        else:
            x = self.x[idx]
            x_val = x[:, 0:2]
            x_mark = x[:, 2:]
            y = self.y[idx]
            # 样本级记忆库所需的稳定ID
            sample_id = np.int64(self.id_offset + idx)

            return (
                x_val.astype(np.float32),
                x_mark.astype(np.float32),
                y.astype(np.float32),
                sample_id,
            )
        
    def custom_collate_fn(self, batch):
        x, x_mark, y, ids = zip(*batch)
        # ids 是 int64，default_collate 会把它们拼成 LongTensor
        ids = [np.int64(i) for i in ids]
        return (
            default_collate(x),
            default_collate(x_mark),
            default_collate(y),
            default_collate(ids),
        )


# class TimeSeriesDataset(Dataset):
#     def __init__(self, x, y, config):
#         self.config = config
#         self.x = x  # x 是输入数据，假设是时间序列
#         self.y = y  # y 是目标数据
#         self.window_size = 360  # 时间窗口大小
#         self.forecast_horizon = 72  # 预测步长
#         self.id_offset = int(getattr(config, "id_offset", 0))

#     def __len__(self):
#         return len(self.x) - self.window_size  # 确保返回的样本数是数据量减去窗口大小

#     def __getitem__(self, idx):
#         # 获取一个时间窗口的数据
#         x_window = self.x[idx: idx + self.window_size]  # 取出时间窗口内的数据
#         x_mark = x_window[:, 2:]  # 提取时间戳或其他特征（从第3列开始）
        
#         # 获取未来72步的数据作为目标值
#         y = np.array(self.y[idx + self.window_size: idx + self.window_size + self.forecast_horizon], dtype=np.float32)  # 确保 y 是 numpy 数组

#         # 样本级记忆库所需的稳定ID
#         sample_id = np.int64(self.id_offset + idx)

#         return (
#             x_window.astype(np.float32),
#             x_mark.astype(np.float32),
#             y,  # 现在 y 已经是 np.float32 类型
#             sample_id,
#         )


#     def custom_collate_fn(self, batch):
#         x, x_mark, y, ids = zip(*batch)
#         # ids 是 int64，default_collate 会把它们拼成 LongTensor
#         ids = [np.int64(i) for i in ids]
#         return (
#             default_collate(x),
#             default_collate(x_mark),
#             default_collate(y),
#             default_collate(ids),
#         )
