# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass

all_batch_size = 32

@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = all_batch_size
    rank: int = 50
    device: str = 'cuda'
    epochs: int = 200
    patience: int = 50
    verbose: int = 10
    try_exp: int = 1
    # density: float = 0.8

