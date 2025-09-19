# Author  :   mkw
# Time    :   2025/09/18 21:30:18
# Desc    :   DAN nn.Module version (forward-only, optimizer handled outside)

import math
import torch
import torch.nn as nn

class DAN(nn.Module):
    """
    统一到外层训练框架的模块化版本：
    - 不在内部创建优化器
    - forward(x, future_feats) 做一次完整前向
    - 其余数据准备（GMM 概率、反归一化、残差回加等）在外部完成
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EncoderLSTM(config)
        self.decoder = DecoderLSTM(config)

    
    def forward(self, x, x_mark=None):
       
        # 编码
        encoder_h, encoder_c, ww = self.encoder(x)

        B = x.size(0)
        T_out = self.config.pred_len
        future_feats = torch.zeros(B, T_out, 2, device=x.device)  # 2表示cos/sin两个特征

        # 解码（future_feats 作为 decoder 的输入）
        out = self.decoder(future_feats, encoder_h, encoder_c, ww)  # [B, T_out]
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False  # 修正：requires_grad
        position = torch.arange(0, max_len).float().unsqueeze(1)   # [max_len,1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 按输入时序长度截取位置编码
        return self.pe[:, : x.size(1)]  # [1, T, d_model]


class EncoderLSTM(nn.Module):
    """
    - 取 x[:, :, 0:2] 作为三个并行 LSTM 的输入
    - 用 x[:, :, 2:5] 与 x[:, :, 5:8] 生成 3 通道 gating 权重的前体，并经注意力+线性+softmax 得到 ww: [B,T_out,3]
    - 返回每个分支的 (hn, cn) 以及 ww
    """
    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt.d_model
        self.layer_dim  = opt.layer
        self.seq_len  = opt.seq_len
        self.pred_len = opt.pred_len
        self.seq_w      = opt.seq_weight
        atten_dim       = opt.atten_dim

        # 三个 LSTM 分支，输入通道数=2
        self.lstm0 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)
        self.lstm1 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)

        # 注意力（batch_first=True 对齐 (B,T,E)）
        self.attn0 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)
        self.attn1 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)
        self.attn4 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)
        self.attn5 = nn.MultiheadAttention(atten_dim, 4, batch_first=True)

        # 线性层
        self.L_out10 = nn.Linear(atten_dim, 1)
        self.L_out11 = nn.Linear(atten_dim, 1)
        self.L_out12 = nn.Linear(atten_dim, 1)
        self.L_out0  = nn.Linear(1, atten_dim // 2)
        self.L_out1  = nn.Linear(1, atten_dim // 2)
        self.L_out2  = nn.Linear(1, atten_dim // 2)

        # 用 LayerNorm 替代 BN，避免 [B,T,E] 维度对不齐问题
        self.norm = nn.LayerNorm(atten_dim)

        # 位置编码（与 half dim 对齐后拼接）
        self.ebb = PositionalEmbedding(atten_dim // 2, self.pred_len)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax_last = nn.Softmax(dim=2)

    def forward(self, x):
        B, L, C = x.shape
        device = x.device
        # ===== 保证至少有 8 个通道 =====
        if C < 8:
            # 复制最后一个通道，补足到 8 通道
            pad = x[:, :, -1:].repeat(1, 1, 8 - C)
            x = torch.cat([x, pad], dim=2)   # [B, L, 8]
            C = 8
        # LSTM 输入通道：0:2
        x0 = x[:, :, 0:2]  # [B,L,2]

        # gating 前体：来自 2:5、5:8 的三通道组装
        ww0 = x[:, :, 5:8]  # [B,L,3]
        ww1 = x[:, :, 2:5]  # [B,L,3]
        ww  = ww0 + ww1 * self.seq_w
        ww  = ww[:, -self.pred_len:, :]  # 对齐到预测窗口长度 [B,T_out,3]

        # 位置编码
        wwe = self.ebb(ww)                 # [1,T_out,d/2]
        wwe = wwe.repeat(B, 1, 1)          # [B,T_out,d/2]

        # 三个分量分别经过：线性升维(到 d/2) → 拼位置编码(到 d) → 两次 self-attn + 残差+norm → 映射到1维
        # comp 0
        z0  = self.tanh(self.L_out0(ww[:, :, 0:1]))      # [B,T,d/2]
        z0  = torch.cat([z0, wwe], dim=2)                # [B,T,d]
        a0, _ = self.attn0(z0, z0, z0)
        z0   = self.norm(z0 + a0)
        a0, _ = self.attn3(z0, z0, z0)
        z0   = self.norm(z0 + a0)
        z0   = self.L_out10(self.relu(z0))               # [B,T,1]

        # comp 1
        z1  = self.tanh(self.L_out1(ww[:, :, 1:2]))      # [B,T,d/2]
        z1  = torch.cat([z1, wwe], dim=2)                # [B,T,d]
        a1, _ = self.attn1(z1, z1, z1)
        z1   = self.norm(z1 + a1)
        a1, _ = self.attn4(z1, z1, z1)
        z1   = self.norm(z1 + a1)
        z1   = self.L_out11(self.relu(z1))               # [B,T,1]

        # comp 2
        z2  = self.L_out2(ww[:, :, 2:3])                 # [B,T,d/2]
        z2  = torch.cat([z2, wwe], dim=2)                # [B,T,d]
        a2, _ = self.attn2(z2, z2, z2)
        z2   = self.norm(z2 + a2)
        a2, _ = self.attn5(z2, z2, z2)
        z2   = self.norm(z2 + a2)
        z2   = self.L_out12(self.relu(z2))               # [B,T,1]

        # 拼成 [B,T,3] 并做 softmax
        ww = torch.cat([z0, z1, z2], dim=2)              # [B,T,3]
        ww = self.softmax_last(ww)                       # [B,T,3]，每步三路权重和=1

        # 三个 LSTM 分支的初始状态（零）
        h0 = torch.zeros(self.layer_dim, B, self.hidden_dim, device=device)
        c0 = torch.zeros(self.layer_dim, B, self.hidden_dim, device=device)

        h, c = [], []
        _, (hn, cn) = self.lstm0(x0, (h0, c0)); h.append(hn); c.append(cn)
        _, (hn, cn) = self.lstm1(x0, (h0, c0)); h.append(hn); c.append(cn)
        _, (hn, cn) = self.lstm2(x0, (h0, c0)); h.append(hn); c.append(cn)

        return h, c, ww  # h/c: 列表长度=3；每个 [num_layers, B, hidden_dim]


class DecoderLSTM(nn.Module):
    """
    三分支 LSTM，用 encoder 的 (h,c) 初始化；用 ww 对三路输出做加权融合。
    """
    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt.d_model
        self.layer_dim  = opt.layer
        self.pred_len = opt.pred_len
        self.tanh = nn.Tanh()

        self.lstm0 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)
        self.lstm1 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(2, self.hidden_dim, self.layer_dim, dropout=0.1,
                             bidirectional=False, batch_first=True)

        self.L_out3 = nn.Linear(self.hidden_dim, 1)
        self.L_out4 = nn.Linear(self.hidden_dim, 1)
        self.L_out5 = nn.Linear(self.hidden_dim, 1)

    def forward(self, future_feats, encoder_h, encoder_c, ww):
        """
        future_feats: [B, T_out, 2]  （cos/sin）
        encoder_h/encoder_c: list of 3 tensors, each [num_layers, B, hidden_dim]
        ww: [B, T_out, 3]
        return: [B, T_out]
        """
        # 取三路初始状态
        h0, h1, h2 = encoder_h
        c0, c1, c2 = encoder_c

        # 三路 LSTM 前向
        out0, _ = self.lstm0(future_feats, (h0, c0))  # [B,T,H]
        out0 = self.L_out3(self.tanh(out0)).squeeze(-1)  # [B,T]

        out1, _ = self.lstm1(future_feats, (h1, c1))
        out1 = self.L_out4(self.tanh(out1)).squeeze(-1)

        out2, _ = self.lstm2(future_feats, (h2, c2))
        out2 = self.L_out5(self.tanh(out2)).squeeze(-1)

        # 加权融合
        w0 = ww[:, :, 0]   # [B,T]
        w1 = ww[:, :, 1]
        w2 = ww[:, :, 2]
        out = out0 * w0 + out1 * w1 + out2 * w2  # [B,T]
        return out.unsqueeze(-1)
