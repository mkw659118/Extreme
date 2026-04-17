# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: ExtremeLSTMMemoConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class ExtremeLSTMMemoConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'extreme_lstm_memo'
    bs: int = 256
    d_model: int = 256
    epochs: int = 200
    patience: int = 40
    dropout: float = 0.3
    e_layers: int = 2
    d_layers: int = 2
    outf: str = './output'
    data_model: str = 'Almaden'
    name: str = 'Almaden'
    seq_len: int = 360
    label_len: int = 180
    pred_len: int = 8
    c_in: int = 3
    
    
   

