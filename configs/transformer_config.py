
from default_config import *
from model_config import OtherConfig


@dataclass
class transformer_config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'transformer'
    bs: int = 32
    d_model: int = 128
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    n_heads: int = 8
    revin: bool = True