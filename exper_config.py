# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 50
    device: str = 'cuda'
    epochs: int = 200
    patience: int = 50
    verbose: int = 10
    try_exp: int = 1
    # density: float = 0.8

    seq_len: int = 96
    pred_len: int = 96



@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    device: str = 'mps'
    bs: int = 32
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

    seq_len: int = 96
    pred_len: int = 96

