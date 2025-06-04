
from default_config import *
from dataclasses import dataclass

from model_config import OtherConfig


@dataclass
class mlp_config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    revin: bool = True