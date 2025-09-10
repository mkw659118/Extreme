# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np

class TensorDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        if self.config.multi_dataset:
            x = self.x[s_begin:s_end][:, :, -3:]
            x_fund = self.x[s_begin:s_end][:, :, 0]
            x_mark = self.x[s_begin:s_end][:, :, 1:4] 
            x_features = self.x[s_begin:s_end][:, :, 4:-3]
            y = self.y[r_begin:r_end]
            # print(f"x.shape = {x.shape}, x_mark.shape = {x_mark.shape}, x_fund.shape = {x_fund.shape}, x_features.shape = {x_features.shape}, y.shape = {y.shape}")
            return x, x_mark, x_fund, x_features, y
        else:
            x = self.x[s_begin:s_end][:, -3:]
            x_fund = self.x[s_begin:s_end][:, 0]
            x_mark = self.x[s_begin:s_end][:, :-1] if not self.config.dataset == 'financial' else self.x[s_begin:s_end][:, 1:-1]
            y = self.y[r_begin:r_end]
            return x, x_mark, x_fund, y

        return False
    

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, x_fund, x_features, y = zip(*batch)
        x, y = default_collate(x), default_collate(y)
        x_mark = default_collate(x_mark)
        x_features = default_collate(x_features)
        x_fund = default_collate(x_fund).long()
        return x, x_mark, x_fund, x_features, y


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
        
    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, y = zip(*batch)
        return default_collate(x), default_collate(x_mark), default_collate(y)
