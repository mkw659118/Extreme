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
    d_model: int = 48
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    n_heads: int = 4
    revin: bool = True
    dropout: float = 0.1



