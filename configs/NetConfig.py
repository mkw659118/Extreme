# coding: utf-8
# Author: mkw
# Date: 2025-06-10 15:45
# Description: ExtremeLSTMMemoConfig
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class NetConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'net'
    bs: int = 64
    d_model: int = 64
    epochs: int = 200
    patience: int = 40
    dropout: float = 0.3
    e_layers: int = 2
    d_layers: int = 2
    outf: str = './output'
    data_model: str = 'Almaden'
    name: str = 'Almaden'
    seq_len: int = 96
    label_len: int = 0
    pred_len: int = 96
    c_in: int = 1
    enc_in: int = 1
    dec_in: int = 1
    out_dim: int = 1
    use_retrieval: bool = True
    use_state_prior: bool = True
    use_missing_aware_encoding: bool = True
    state_num: int = 0
    num_experts: int = 4
    top_k_experts: int = 2
    retrieval_num: int = 2
    state_prior_distribution: str = 'student_t'
    state_prior_scales: str = '1,4,8,16'
    state_prior_include_seq_level: bool = True
    
    
   

