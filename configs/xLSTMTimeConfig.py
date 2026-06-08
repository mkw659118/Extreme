
from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class xLSTMTimeConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'xLSTMTime'
    bs: int = 128
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    d_model: int = 64
    channel: int = 7
    embedding_dim: int = 256
    patch_size: int = 16
    stride: int = 8
    num_heads: int = 4
    conv1d_kernel_size: int = 4
    num_blocks: int = 2
   