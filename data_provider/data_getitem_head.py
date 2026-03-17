#Author  :   mkw 
#Time    :   2025/09/30 14:23:08
#Desc    :   None

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, config, route_labels=None):
        self.config = config
        self.x = x
        self.y = y
        self.route_labels = route_labels  # 新增 route_labels
        self.id_offset = int(getattr(config, "id_offset", 0))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        x_mark = x[:, 1:]
        y = self.y[idx]
        sample_id = np.int64(self.id_offset + idx)

        if self.route_labels is None:
            route_label = np.int64(-1)
        else:
            route_label = np.int64(self.route_labels[idx])

        return (
            x.astype(np.float32),
            x_mark.astype(np.float32),
            y.astype(np.float32),
            sample_id,
            route_label,  # 返回 route_label
        )

    def custom_collate_fn(self, batch):
        x, x_mark, y, ids, route_labels = zip(*batch)
        ids = [np.int64(i) for i in ids]
        route_labels = [np.int64(r) for r in route_labels]
        return (
            default_collate(x),
            default_collate(x_mark),
            default_collate(y),
            default_collate(ids),
            default_collate(route_labels),  # 返回 route_labels
        )

# class TimeSeriesDataset(Dataset):
#     def __init__(self, x, y, config):
#         self.config = config
#         self.x = x
#         self.y = y
#         self.id_offset = int(getattr(config, "id_offset", 0))

#     def __len__(self):
#         return len(self.x)
       
#     def __getitem__(self, idx): 
#         x = self.x[idx]
#         x_mark = x[:, 2:]
#         y = self.y[idx]
#         # 样本级记忆库所需的稳定ID
#         sample_id = np.int64(self.id_offset + idx)

#         return (
#             x.astype(np.float32),
#             x_mark.astype(np.float32),
#             y.astype(np.float32),
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



# class TimeSeriesDataset(Dataset):
#     def __init__(self, x, y, config):
#         self.config = config
#         self.x = x
#         self.y = y
#         self.id_offset = int(getattr(config, "id_offset", 0))

#     def __len__(self):
#         return len(self.x)
#         # return len(self.x) - self.config.seq_len - self.config.pred_len + 1

#     def __getitem__(self, idx):
#         s_begin = idx
#         s_end = s_begin + self.config.seq_len
#         r_begin = s_end
#         r_end = r_begin + self.config.pred_len

#         x = self.x[s_begin:s_end]
#         x_mark = self.x[s_begin:s_end]
#         y = self.y[r_begin:r_end]
#         sample_id = np.int64(self.id_offset + idx)
#         return x, x_mark, y, sample_id

#     def custom_collate_fn(self, batch):
#         from torch.utils.data.dataloader import default_collate
#         x, x_mark, y, ids = zip(*batch)
#         x, y = default_collate(x), default_collate(y)
#         x_mark = default_collate(x_mark)
#         ids = default_collate(ids)

#         return x, x_mark, y, ids