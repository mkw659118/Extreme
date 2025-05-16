# coding : utf-8
# Author : Yuxiang Zeng
import torch

from baselines.timesnet import TimesBlock
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.feedforward.moe import MoE
from layers.revin import RevIN
from layers.transformer import Transformer

class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft
        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)

        self.projection = torch.nn.Linear(enc_in, config.rank, bias=True)
        # self.projection = TokenEmbedding(enc_in, config.rank)
        self.position_embedding = PositionEncoding(d_model=self.rank, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(999999, config.rank)
        # self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.encoder = Transformer(
            self.rank,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )

        self.encoder2 = Transformer(
            self.rank,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )
        # self.moe = MoE(d_model=self.rank, d_ff=self.rank, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2, loss_coef=0.001)
        # self.moe = MoE(d_model=self.rank * 33, d_ff=self.rank * 33, num_m=1, num_router_experts=8, num_share_experts=1, num_k=2, loss_coef=0.001)

        self.decoder = torch.nn.Linear(config.rank, 1)
        # self.moe = SparseMoE(self.rank * (config.seq_len + config.pred_len), self.rank * (config.seq_len + config.pred_len), 8, noisy_gating=True, num_k=1, loss_coef=1e-3)
        # self.encoder = torch.nn.ModuleList([TimesBlock(config) for _ in range(config.num_layers)])
        # self.layer_norm = torch.nn.LayerNorm(self.rank)

    def forward(self, x, x_mark, x_fund):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        # print(x.shape, x_mark.shape)
        x_enc = self.projection(x)
        # x_enc += self.position_embedding(x_enc)
        # x_enc += self.fund_embedding(x_fund)
        # x_enc += self.temporal_embedding(x_mark)

        # 对齐时间维度
        # [bs, seq, chanel, d] -> [bs, d, seq] -> [bs, d, pred] -> [bs, pred, d]
        x_enc = self.predict_linear(x_enc.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # align temporal dimension
        # [bs, seq, d] -> [bs, d, seq] -> [bs, d, pred] -> [bs, pred, d]
        # x_enc = self.predict_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        # ===== 1. 跨通道 attention =====
        # 原始 x_enc: (batch, time, channel, dim)
        # 调整为 (channel, batch*time, dim)
        bs, pred_len, channels, dim = x_enc.shape
        x_enc = x_enc.permute(2, 0, 1, 3).reshape(channels, bs * pred_len, dim)
        x_enc = self.encoder(x_enc)
        # 还原为 (batch, time, channel, dim)
        x_enc = x_enc.reshape(channels, bs, pred_len, dim).permute(1, 2, 0, 3)

        # ===== 2. 跨时间 attention =====
        # 调整为 (time, batch*channel, dim)
        x_enc = x_enc.permute(1, 0, 2, 3).reshape(pred_len, bs * channels, dim)
        # 注意力：时间步之间的 self-attention
        x_enc = self.encoder2(x_enc)
        # 还原为 (batch, time, channel, dim)
        x_enc = x_enc.reshape(pred_len, bs, channels, dim).permute(1, 0, 2, 3)

        # x_enc = self.encoder(x_enc)

        # MoE
        # bs, now_len, d_model = x_enc.shape
        # x_enc = torch.reshape(x_enc, (bs, now_len * d_model))
        # x_enc, aux_loss = self.moe(x_enc)
        # x_enc = torch.reshape(x_enc, (bs, now_len, d_model))

        # x_enc = self.moe(x_enc)
        # self.aux_loss = self.moe.aux_loss

        # TimesNet
        # for i in range(len(self.encoder)):
        #     x_enc = self.layer_norm(self.encoder[i](x_enc))
        # x_enc += torch.cat([self.fund_embedding(code_idx), self.fund_embedding(code_idx)], dim=1)
        # y = self.fc(x_enc.reshape(x_enc.size(0), -1))

        x_enc = self.decoder(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]

        # denorm
        if self.revin:
            y = self.revin_layer(y, 'denorm')
            y = y[:, :, -1]  # [B, L, 1]
        return y