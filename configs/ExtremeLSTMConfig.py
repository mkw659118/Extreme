# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: ExtremeLSTMConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class ExtremeLSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'extreme_lstm'
    bs: int = 256
    d_model: int = 32
    epochs: int = 200
    patience: int = 40
    n_heads: int = 4
    revin: bool = False
    dropout: float = 0.3
    win_size: int = 16
    patch_len: int = 16
    use_memory: bool = True
    num_layers_intra_patch: int = 2
    num_layers_inter_patch: int = 2
    outf: str = './output'
    data_model: str = 'Almaden'
    name: str = 'Almaden'
    mem_mode: str = 'retrieval'
    seq_weight: float = 0.4
    d_ff: int = 1024
    e_layers: int = 1
    seq_len: int = 360
    pred_len: int = 72
    
    
   

