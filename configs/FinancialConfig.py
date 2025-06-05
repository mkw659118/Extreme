
from dataclasses import dataclass
from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class FinancialConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 56
    epochs: int = 200
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 45
    verbose: int = 5
    try_exp: int = 1
    dataset: str = 'financial'  # financial  weather
    scaler_method: str = 'global'
    multi_dataset: bool = True

    seq_len: int = 16
    pred_len: int = 10
    input_size: int = 3
    

    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'cosine'
    fft: bool = False
    revin: bool = False
    idx: int = 0

    # start_date: str = '2022-07-13'
    # end_date: str = '2023-07-13'
    start_date: str = '2020-05-15'
    end_date: str = '2025-05-15'