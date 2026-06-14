# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    bs: int = 128
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'  # L1Loss  MSELoss
    optim: str = 'Adam'
    epochs: int = 200
    patience: int = 40
    verbose: int = 5
    device: str = 'cuda'
    monitor_metric: str = 'RMSE'
    use_amp: bool = False
    mode: str = 'train'
    use_prior_guided_tail_residual: bool = True
    tail_gate_scale: float = 1.0
    use_tail_aware_loss: bool = True
    tail_loss_lambda1: float = 1.0
    tail_loss_lambda2: float = 0.5
    tail_loss_lambda3: float = 0.1
    tail_mask_quantile: float = 0.9
    state_balance_weight: float = 0.02
    state_dom_cap: float = 0.8
    freeze_prior_after_pretrain: bool = False
    state_prior_temperature_start: float = 1.0
    state_prior_temperature_end: float = 0.6


@dataclass
class BaseModelConfig:
    model: str = 'patch_extreme_memory_transformer'
    rank: int = 40
    retrain: bool = True
    num_layers: int = 3
    


@dataclass
class DatasetInfo:
    path: str = './datasets'
    dataset: str = 'Abilene'
    train_size: int = 500
    use_train_size: bool = False
    density: float = 0.70
    eval_set: bool = True
    shuffle: bool = False
    scaler_method: str = 'stander'
    # reservoir_sensor: str = 'reservoir_stor_4007_sof24'
    reservoir_sensor: str = 'reservoir_stor_4001_sof24'
    # start_point: str = '1983-07-01 23:30:00'
    start_point: str = '1991-07-01 23:30:00'
    train_end: str = '2018-06-30 23:30:00'
    train_volume: int = 40000
    val_size: int = 60
    test_start: str = '2018-07-01 00:30:00'
    test_end: str = '2019-07-01 00:30:00'
    oversampling: int = 40
    os_s: int = 18
    os_v: int = 4
    # val_seed: int = 2007
    # train_seed: int = 1010
    val_seed: int = 2025
    train_seed: int = 2024
    roll: int = 8
    enc_in: int = 1
    data_file: str = 'Abilene_single.csv'
    target_col: int = 0
    artificial_missing_rate: float = 0.0
    artificial_missing_seed: int = 2026
    artificial_missing_splits: str = 'train,val,test'
    artificial_missing_target_only: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 1
    debug: bool = False
    record: bool = True
    hyper_search: bool = False
    continue_train: bool = False
    outf: str = './output'
    data_model: str = 'Almaden'


@dataclass
class LoggerConfig:
    logger: str = 'mkw'
