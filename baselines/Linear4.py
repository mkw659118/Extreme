# coding: utf-8
# Author: mkw
# Date: 2025-06-05 18:35
# Description: Linear4
from einops import rearrange
from torch import nn

from layers.revin import RevIN


# 以下是做法三：timesNet

class Linear4(nn.Module):
    def __init__(self, enc_in, config):
        super(Linear4, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.d_model = config.d_model           # 中间维度（类似 hidden size）

        # 可选的可逆归一化层（RevIN），用于在归一化后恢复原始尺度
        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        # 中间线性层，对时间维度进行映射（注意是对 seq_len 映射到更大维度）
        # 输入维度：seq_len → 输出维度：d_model + seq_len
        self.middle_linear = nn.Linear(self.seq_len, self.seq_len + self.seq_len)

        # 用于将中间表示直接预测为 pred_len 步（这层在 forward 中未使用）
        self.predict_linear = nn.Linear(self.seq_len + self.seq_len, self.pred_len)

        # 对特征维度进行非线性特征提升（例如 21 → 50）
        self.up_feature_linear = nn.Linear(self.config.input_size, self.d_model)

        # 再降回原特征维度（例如 50 → 21）
        self.down_feature_linear = nn.Linear(self.d_model, self.config.input_size)

    def forward(self, x, x_mark):
        """
        输入:
            x: [B, L, D]，原始输入序列，B=batch_size, L=seq_len, D=特征维度
            x_mark: 时间信息等辅助变量（此模型中未使用）

        输出:
            y: [B, pred_len, D]，预测结果（仅保留最后 pred_len 步）
        """

        # 1. 可选的可逆归一化，提升训练稳定性
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # 2. 交换顺序，将时间维度调到最后一维以便对时间维度做 linear 投影
        #    输入 [B, L, D] → 输出 [B, D, L]
        x = rearrange(x, 'B L D -> B D L')

        # 3. 对seq_len做 Linear 映射：
        #    输入 [B, D, seq_len] → 输出 [B, D, d_model + seq_len]
        y = self.middle_linear(x)

        # 4. 交换维度，将中间维度置于中间，以便下游 Linear 操作：
        #    输入 [B, D, d_model + seq_len] → 输出 [B, d_model + seq_len, D]
        y = rearrange(y, 'b d D_model -> b D_model d')

        # 5. 对特征维度做升维（D → 50）
        y = self.up_feature_linear(y)  # [B, d_model + seq_len, 50]

        # 6. 降回原始特征维度（50 → D）
        y = self.down_feature_linear(y)  # [B, d_model + seq_len, D]

        # 7. 保留时间维度上的最后 pred_len 步作为预测结果
        y = y[:, -self.pred_len:, :]  # [B, pred_len, D]

        # 8. 可选的反归一化（如果启用了 RevIN）
        if self.revin:
            y = self.revin_layer(y, 'denorm')

        return y  # [B, pred_len, D]
