# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    bs: int = 256
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'  # L1Loss  MSELoss
    optim: str = 'Adam'
    epochs: int = 10
    patience: int = 3
    verbose: int = 10
    device: str = 'cuda'
    monitor_metric: str = 'MAE'
    use_amp: bool = False


@dataclass
class BaseModelConfig:
    model: str = 'ours'
    rank: int = 40
    retrain: bool = True
    num_layers: int = 3


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'water'
    train_size: int = 500
    use_train_size: bool = False
    density: float = 0.70
    eval_set: bool = True
    shuffle: bool = False
    scaler_method: str = 'stander'
    spliter_ratio: str = '7:1:2'
    reservoir_sensor: str = 'reservoir_stor_4001_sof24'
    start_point: str = '1991-07-01 23:30:00'
    train_end: str = '2018-06-30 23:30:00'
    train_volume: int = 40000
    val_size: int = 60
    test_start: str = '2018-07-01 00:30:00'
    test_end: str = '2019-07-01 00:30:00'
    oversampling: int = 40
    os_s: int = 18
    os_v: int = 4
    split_mode: str = 'ds'
    val_seed: int = 2007
    train_seed: int = 1010





@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 1
    debug: bool = False
    record: bool = True
    hyper_search: bool = False
    continue_train: bool = False


@dataclass
class LoggerConfig:
    logger: str = 'mkw'