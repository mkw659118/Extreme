# coding: utf-8
# Author: mkw
# Date: 2025-06-08 14:37
# Description: xlstm_mixerConfig

from configs.default_config import *
from dataclasses import dataclass

from configs.MainConfig import OtherConfig


@dataclass
class xlstm_mixerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'xlstm_mixer'
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
    moving_avg: int = 25
    task_name: str = 'long_term_forecast'
    c_out: int = 1
    dropout: float = 0.1
    patch_len: int = 16
    type: str = 'cuda'
    # Portable default; "cuda" requires Ninja and a local C++/CUDA compiler.
    slstm_backend: str = 'vanilla'
      
    
