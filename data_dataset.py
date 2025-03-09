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
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y


def custom_collate_fn(batch, config):
    from torch.utils.data.dataloader import default_collate
    x, y = zip(*batch)
    x, y = default_collate(x), default_collate(y)
    return x, y