# coding : utf-8
# Author : Yuxiang Zeng
import torch

from baselines.timesnet import TimesBlock
from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.feedforward.smoe import SparseMoE
from layers.revin import RevIN
from layers.transformer import Transformer
from modules.pretrain_timer import Timer
from modules.temporal_enc import TemporalEmbedding


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

        # self.projection = torch.nn.Linear(1, config.rank, bias=True)
        self.projection = TokenEmbedding(1, config.rank)
        self.position_embedding = PositionEncoding(d_model=self.rank, max_len=config.seq_len, method='bert')
        self.fund_embedding = torch.nn.Embedding(999999, config.rank)
        self.temporal_embedding = TemporalEmbedding(self.rank, 'embeds')
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.encoder = Transformer(
            self.rank,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )

        self.moe = SparseMoE(self.rank * config.seq_len * 2, self.rank * config.seq_len, 8, noisy_gating=True, num_k=2, loss_coef=1e-3)
        # self.encoder = torch.nn.ModuleList([TimesBlock(config) for _ in range(config.num_layers)])
        # self.layer_norm = torch.nn.LayerNorm(self.rank)
        self.fc = torch.nn.Linear(config.rank, 1)

        # Pretrain Timer
        # self.backbone = Timer(config)
        # self.enc_embedding = self.backbone.patch_embedding
        # self.decoder = self.backbone.decoder
        # # self.proj = self.backbone.proj
        # self.proj = torch.nn.Linear(1024, config.pred_len)
        # ckpt_path = 'Timer_forecast_1.0.ckpt'
        # sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
        # sd = {k[6:]: v for k, v in sd.items()}
        # try:
        #     self.backbone.load_state_dict(sd, strict=True)
        #     print("✅ Backbone state_dict 加载成功。")
        # except RuntimeError as e:
        #     print("❌ 加载失败：", e)
        # exit()

    def forward(self, x, x_mark):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        x = x.unsqueeze(-1)

        # print(x.shape, x_mark.shape)
        x_enc = self.projection(x)
        x_enc += self.position_embedding(x_enc)
        # x_enc += self.tempora l_embedding(x_mark)
        # x_enc += self.fund_embedding(code_idx)
        x_enc = self.predict_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        x_enc = self.encoder(x_enc)

        # MoE
        bs, now_len, d_model = x_enc.shape
        x_enc = torch.reshape(x_enc, (bs, now_len * d_model))
        x_enc, aux_loss = self.moe(x_enc)
        self.aux_loss = aux_loss
        x_enc = torch.reshape(x_enc, (bs, now_len, d_model))

        # for i in range(len(self.encoder)):
        #     x_enc = self.layer_norm(self.encoder[i](x_enc))

        # x_enc += torch.cat([self.fund_embedding(code_idx), self.fund_embedding(code_idx)], dim=1)

        x_enc = self.fc(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]
        # y = self.fc(x_enc.reshape(x_enc.size(0), -1))

        # denorm
        if self.revin:
            y = self.revin_layer(y, 'denorm')
        return y

    # def forward(self, x, x_mark):
    #     x = x.unsqueeze(-1)
    #     B, L, M = x.shape
    #     x_enc = x
    #
    #     # Normalization from Non-stationary Transformer
    #     # means = x_enc.mean(1, keepdim=True).detach()
    #     # x_enc = x_enc - means
    #     # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    #     # x_enc /= stdev
    #
    #     # Step 1: 交换时间维度和变量维度
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # 输入: x_enc shape: [B, T, M]
    #     # 输出: x_enc shape: [B, M, T]
    #     # 解释: B是batch size，T是时间长度，M是变量数。这里将变量维度提前，为后续patching按变量维度做操作。
    #
    #     # Step 2: 进入Embedding模块
    #     dec_in, n_vars = self.enc_embedding(x_enc)
    #     # 输入: [B, M, T]
    #     # 输出: dec_in shape: [B * M, N, D]，n_vars = M
    #     # 解释: 将每个变量视为一个序列，每个序列被切成N个patch，每个patch变为D维embedding。共B个样本，每个样本有M个变量，所以展开为 B*M 个序列。
    #
    #     # Step 3: Transformer编码
    #     dec_out, attns = self.decoder(dec_in)
    #     # 输入: dec_in shape: [B * M, N, D]
    #     # 输出: dec_out shape: [B * M, N, D]
    #     # 解释: 标准的Transformer block输出，attns 是注意力得分，可以用于可视化或中间提取。
    #
    #     # Step 4: 输出维度映射
    #     dec_out = self.proj(dec_out)
    #     # 输入: [B * M, N, D]
    #     # 输出: [B * M, N, L]
    #     # 解释: proj 是一个线性层，将每个token的embedding从 D 映射到预测长度 L（可能是时间片段长度）。
    #
    #     # Step 5: reshape 为最终输出形式
    #     dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)
    #     # 输入: [B * M, N, L] → reshape 为 [B, M, N * L] → transpose 为 [B, T, M]
    #     # 输出: dec_out shape: [B, T, M]
    #     # 解释: 每个变量对应的 patch 被拼接为完整的时间序列，再把变量维度放到最后，恢复为常规时间序列格式。
    #
    #     # De-Normalization from Non-stationary Transformer
    #     # dec_out = dec_out * stdev + means
    #
    #     return dec_out.squeeze(-1)
