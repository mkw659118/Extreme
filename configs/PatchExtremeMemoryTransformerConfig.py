# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: PatchExtremeMemoryTransformerConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class PatchExtremeMemoryTransformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'patch_extreme_memory_transformer'
    bs: int = 256
    d_model: int = 256
    epochs: int = 50
    patience: int = 10
    n_heads: int = 4
    revin: bool = True
    dropout: float = 0
    win_size: int = 16
    patch_len: int = 16
    use_memory: bool = True
    num_layers_intra_patch: int = 1
    num_layers_inter_patch: int = 1
    outf: str = './output'
    data_model: str = 'Almaden'
    # data_model: str = 'Lexington'
    name: str = 'Almaden'
    # name: str = 'Lexington'
    mem_mode: str = 'tbm'
    seq_weight: float = 0.4
    momentum: float = 0.05
    r: int = 1
    lambda_div: float = 1e-2
   

