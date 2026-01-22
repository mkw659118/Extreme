# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: DLinearConfig

from configs.default_config import *
from dataclasses import dataclass

from configs.MainConfig import OtherConfig


@dataclass
class DLinearConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'DLinear'
    bs: int = 256
    rank: int = 32
    epochs: int = 200
    patience: int = 40
    verbose: int = 100
    num_layers: int = 2
    revin: bool = False
    d_model: int = 64
    kernel_size: int = 25
    individual: bool = True
    
