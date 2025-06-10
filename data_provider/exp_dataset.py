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

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        if self.config.model == 'transformer_library':
            # Decoder输入从Encoder结尾前label_len个时间步开始
            r_begin = s_end - self.config.label_len
            r_end = s_end + self.config.pred_len

            x_enc = self.x[s_begin:s_end][:, 4:]
            x_mark_enc = self.x[s_begin:s_end][:, :4]
            x_dec = self.x[r_begin:r_end][:, 4:]
            x_mark_dec = self.x[r_begin:r_end][:, :4]
            y = self.y[s_end:s_end + self.config.pred_len]
            return x_enc, x_mark_enc, x_dec, x_mark_dec, y
        else:
            x = self.x[s_begin:s_end][:, 4:]
            x_mark = self.x[s_begin:s_end][:, :4]
            y = self.y[r_begin:r_end]
            return x, x_mark, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, x_fund, y = zip(*batch)
        x, y = default_collate(x), default_collate(y)
        x_mark = default_collate(x_mark)
        x_fund = default_collate(x_fund).long()
        return x, x_mark, x_fund, y


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    # def __getitem__(self, idx):
    #     s_begin = idx
    #     s_end = s_begin + self.config.seq_len
    #     r_begin = s_end
    #     r_end = r_begin + self.config.pred_len
    #
    #     x = self.x[s_begin:s_end][:, 4:]
    #     x_mark = self.x[s_begin:s_end][:, :4]
    #     y = self.y[r_begin:r_end]
    #     return x, x_mark, y
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        if self.config.model == 'transformer_library':
            # Decoder输入从Encoder结尾前label_len个时间步开始
            r_begin = s_end - self.config.label_len
            r_end = s_end + self.config.pred_len
            x_enc = self.x[s_begin:s_end][:, 4:]
            x_mark_enc = self.x[s_begin:s_end][:, :4]
            x_dec = self.x[r_begin:r_end][:, 4:]
            x_mark_dec = self.x[r_begin:r_end][:, :4]
            y = self.y[s_end:s_end + self.config.pred_len]
            return x_enc, x_mark_enc, x_dec, x_mark_dec, y
        else:
            x = self.x[s_begin:s_end][:, 4:]
            x_mark = self.x[s_begin:s_end][:, :4]
            y = self.y[r_begin:r_end]
            return x, x_mark, y

    # def custom_collate_fn(self, batch, config):
    #     from torch.utils.data.dataloader import default_collate
    #     x, x_mark, y = zip(*batch)
    #     x, y = default_collate(x), default_collate(y)
    #     x_mark = default_collate(x_mark)
    #     return x, x_mark, y
    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        if config.model == 'transformer_library':
            # 支持五元组解包
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = zip(*batch)
            return (
                default_collate(x_enc),
                default_collate(x_mark_enc),
                default_collate(x_dec),
                default_collate(x_mark_dec),
                default_collate(y)
            )
        else:
            # 默认处理三元组
            x, x_mark, y = zip(*batch)
            return (
                default_collate(x),
                default_collate(x_mark),
                default_collate(y)
            )
