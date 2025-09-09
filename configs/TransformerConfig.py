# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: TransformerConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class TransformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'transformer'
    bs: int = 32
    d_model: int = 128
    epochs: int = 20
    patience: int = 4
    verbose: int = 1
    num_layers: int = 3
    n_heads: int = 4
    revin: bool = True
    dropout: float = 0.1
    amp: bool = True
    match_mode: str = 'abc'
    constraint: bool = False
    win_size: int = 48
    patch_len: int = 16


