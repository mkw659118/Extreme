# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from baselines.CrossFormer.Crossformer import Crossformer
from baselines.Linear import Linear
from baselines.Linear2 import Linear2
from baselines.TimeLLM.TimeLLM import timeLLM
from baselines.TimesNet.TimesNet import TimesNet
from layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from modules.backbone import Backbone
from baselines.encoder_seq import SeqEncoder
from modules.ts_model import TimeSeriesModel


class Model(BasicModel):
    def __init__(self, datamodule, config):
        super().__init__(config)
        self.config = config
        # self.input_size = datamodule.train_loader.dataset.x.shape[-1]
        self.input_size = 1 if config.dataset != 'financial' else 3
        self.hidden_size = config.rank

        if config.model == 'ours':
            self.model = TimeSeriesModel(self.input_size, config)

        # 2025年05月30日11:45:49 这里只使用了一层的Linear，效果：
        elif config.model == 'mlp':
            self.model = Linear(self.input_size, config)
        elif config.model == 'mlp2':
            self.model = Linear2(self.input_size, config)

        elif config.model in ['rnn', 'lstm', 'gru']:
            self.model = SeqEncoder(
                input_size=self.input_size,
                d_model=self.hidden_size,
                seq_len=config.seq_len,
                pred_len=config.seq_len,
                num_layers=config.num_layers,
                seq_method=config.model,
                bidirectional=True
            )
        elif config.model == 'crossformer':  # 添加 Crossformer 支持
            self.model = Crossformer(
                data_dim=self.input_size,
                in_len=config.seq_len,
                out_len=config.pred_len,
                seg_len=config.seg_len,  # 使用 config.seg_len
                win_size=4,
                d_model=self.hidden_size,
                n_heads=8,
                e_layers=2,
                dropout=0.1,
                device=config.device
            )
        elif config.model == 'timesnet':
            self.model = TimesNet(enc_in=self.input_size, configs=config)

        elif config.model == 'timellm':
            self.model = timeLLM(config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")


        if config.multi_dataset:
            self.model = Backbone(self.input_size, config)
            self.distance = PairwiseLoss(method=config.dis_method, reduction='mean')