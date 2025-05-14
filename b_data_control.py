# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset


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
        # x = self.x[idx][:, -1]  # [..., np.newaxis]
        # y = self.y[idx]
        # x_mark = self.x[idx][:, :-1] if not self.config.dataset == 'financial' else self.x[idx][:, 1:-1]
        # x_fund = self.x[idx][:, 0]

        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len
        # print(s_begin, s_end, r_begin, r_end)

        if not self.config.multi_dataset:
            x = self.x[s_begin:s_end][:, -3:]
            x_fund = self.x[s_begin:s_end][:, 0]
            x_mark = self.x[s_begin:s_end][:, :-1] if not self.config.dataset == 'financial' else self.x[s_begin:s_end][:, 1:-1]
            y = self.y[r_begin:r_end]
        else:
            x = self.x[s_begin:s_end][:, :, -3:]
            x_fund = self.x[s_begin:s_end][:, :, 0]
            x_mark = self.x[s_begin:s_end][:, :, -1] if not self.config.dataset == 'financial' else self.x[s_begin:s_end][:, :, 1:-1]
            y = self.y[r_begin:r_end]

        # print(x.shape, y.shape)
        return x, x_mark, x_fund, y

def custom_collate_fn(batch, config):
    from torch.utils.data.dataloader import default_collate
    x, x_mark, x_fund, y = zip(*batch)
    x, y = default_collate(x), default_collate(y)
    x_mark = default_collate(x_mark)
    x_fund = default_collate(x_fund).long()
    return x, x_mark, x_fund, y