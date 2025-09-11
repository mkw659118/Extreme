# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        self.split_mode = getattr(config, "split_mode", "ratio")

    def __len__(self):
        if self.split_mode == "ds":
            # ds_like_split_dataset 已经切好窗口
            return len(self.x)
        else:
            return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    def __getitem__(self, idx):
        if self.split_mode == "ds":
            # 直接返回
            x = self.x[idx]
            x_mark = x[:, :4]
            x_val = x[:, 4:]
            y = self.y[idx]
            return x_val.astype(np.float32), x_mark.astype(np.float32), y.astype(np.float32)
        else:
            # 旧逻辑: 自己滑窗
            s_begin = idx
            s_end = s_begin + self.config.seq_len
            r_begin = s_end
            r_end = r_begin + self.config.pred_len

            x = self.x[s_begin:s_end][:, 4:]
            if x.ndim == 1:
                x = np.expand_dims(x, -1)

            x_mark = self.x[s_begin:s_end][:, :4]
            y = self.y[r_begin:r_end]
            if y.ndim == 1:
                y = np.expand_dims(y, -1)

            return x.astype(np.float32), x_mark.astype(np.float32), y.astype(np.float32)
        
    def custom_collate_fn(self, batch):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, y = zip(*batch)
        return default_collate(x), default_collate(x_mark), default_collate(y)
