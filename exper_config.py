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


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    bs: int = 32
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class RNNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'rnn'
    bs: int = 128
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'
    bs: int = 128
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class GRUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gru'
    bs: int = 128
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
