#Author  :   mkw 
#Time    :   2025/09/30 14:23:08
#Desc    :   None


from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode
        self.split_mode = getattr(config, "split_mode", "ratio")
        # 可选：不同数据划分使用不同偏移，避免 sample_id 冲突（例如 train=0, val=1e9, test=2e9）
        self.id_offset = int(getattr(config, "id_offset", 0))

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
            # 样本级记忆库所需的稳定ID
            sample_id = np.int64(self.id_offset + idx)

            return (
                x_val.astype(np.float32),
                x_mark.astype(np.float32),
                y.astype(np.float32),
                sample_id,
            )
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

            # 每个窗口作为一个样本：用窗口起点 s_begin 作为稳定ID
            sample_id = np.int64(self.id_offset + s_begin)

            return (
                x.astype(np.float32),
                x_mark.astype(np.float32),
                y.astype(np.float32),
                sample_id,
            )

    def custom_collate_fn(self, batch):
        """
        返回四元组：
          x:       [B, seq_len, Cx]  float32
          x_mark:  [B, seq_len, 4]   float32
          y:       [B, pred_len, Cy] float32
          ids:     [B]               int64（LongTensor）
        """
        from torch.utils.data.dataloader import default_collate
        x, x_mark, y, ids = zip(*batch)

        # 注意：确保 ids 是 int64，default_collate 会把它们拼成 LongTensor
        ids = [np.int64(i) for i in ids]

        return (
            default_collate(x),
            default_collate(x_mark),
            default_collate(y),
            default_collate(ids),
        )
