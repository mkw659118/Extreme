# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: TransformerConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class PMDformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'PMDformer'
    bs: int = 256
    d_model: int = 64
    epochs: int = 50
    patience: int = 8
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
    d_ff: int = 512
    embed : str = 'timeF'
    factor: int = 3
    activation: str = 'gelu'
    freq: str = 'h'
    e_layers: int = 2
    d_layers: int = 1
    task_name: str = 'long_term_forecast'  
    c_out : int = 1  
    label_len: int = 48  
    dec_in : int = 1  
    distil: bool = True
    patch_size: int = 24
    v_layers : int = 2
    