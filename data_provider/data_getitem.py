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
