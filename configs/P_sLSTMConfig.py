
from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class P_sLSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'P_sLSTM'
    bs: int = 128
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    d_model: int = 64
