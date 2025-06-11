# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
import torch
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
        # 统一转换为 PyTorch tensor，避免 __getitem__ 重复转换
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        print("ajhahlfahlhfahfakjuuuuuuuuuuuuuuuuuu")
        print(x.shape)
        print(y.shape)
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
        # 序列起始位置：从 idx 开始取样本
        s_begin = idx
        # 编码器输入序列的结束位置（不包含该位置）
        s_end = s_begin + self.config.seq_len
        # 原始的解码器目标（预测段）起止范围
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        # 如果当前模型是 transformer_library 模型，则需要构造 decoder 的输入
        if self.config.model == 'transformer_library':
            label_len = self.config.label_len
            pred_len = self.config.pred_len
            dec_len = label_len + pred_len

            # ------------------- 构造 Encoder 输入 -------------------
            x_enc = self.x[s_begin:s_end][:, 4:]
            x_mark_enc = self.x[s_begin:s_end][:, :4]

            # ------------------- 构造 Decoder 输入 -------------------
            d_actual = self.x.shape[1] - 4  # 真实变量维度
            x_dec = torch.zeros((dec_len, d_actual))
            x_dec[:label_len] = self.x[s_end - label_len:s_end][:, 4:]

            x_mark_dec = self.x[s_end - label_len:s_end + pred_len][:, :4]

            # ------------------- 构造预测目标 -------------------
            y = self.y[s_end:s_end + pred_len]

            return x_enc, x_mark_enc, x_dec, x_mark_dec, y

        else:
            # ------------------- 兼容其他模型（非 Transformer） -------------------
            # x: 输入特征 [seq_len, D]，不使用 decoder，只返回 encoder 输入
            x = self.x[s_begin:s_end][:, 4:]
            # x_mark: 对应时间戳 [seq_len, 4]
            x_mark = self.x[s_begin:s_end][:, :4]
            # y: 预测目标值 [pred_len, D]
            y = self.y[r_begin:r_end]

            return x, x_mark, y

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
