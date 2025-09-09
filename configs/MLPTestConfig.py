
from configs.default_config import *
from dataclasses import dataclass

from configs.MainConfig import OtherConfig


@dataclass
class MLPTestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp_test'
    bs: int = 32
    rank: int = 32
    epochs: int = 10
    patience: int = 3
    verbose: int = 1
    num_layers: int = 2
    revin: bool = True
    d_model: int = 64
