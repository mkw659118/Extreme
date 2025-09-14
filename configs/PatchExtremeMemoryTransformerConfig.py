# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: TransformerConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class PatchExtremeMemoryTransformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'patch_extreme_memory_transformer'
    bs: int = 256
    d_model: int = 64
    epochs: int = 200
    patience: int = 40
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


