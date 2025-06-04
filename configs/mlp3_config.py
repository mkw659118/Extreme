
from default_config import *
from model_config import OtherConfig


@dataclass
class mlp3_config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp3'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    hidden_dim: int = 256
    revin: bool = True