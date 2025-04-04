# coding : utf-8
# Author : yuxiang Zeng

from d_default_config import *
from dataclasses import dataclass

@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 64
    device: str = 'cuda'
    epochs: int = 200
    patience: int = 30
    verbose: int = 1
    try_exp: int = 1
    num_layers: int = 4

    norm_method: str = 'rms'
    ffn_method: str = 'moe'
    att_method: str = 'self'
    dis_method: str = 'dtw'
    # dataset: str = 'financial'
    multi_dataset: bool = False
    idx: int = 0


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2


@dataclass
class CrossformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'crossformer'  
    bs: int = 256
    rank: int = 32
    verbose: int = 1
    seg_len: int = 6


@dataclass
class TimesNetConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timesnet'  # 模型类型
    bs: int = 32  # 批大小
    epochs: int = 10  # 训练周期
    d_model: int = 32
    d_ff: int = 32  # 前馈层大小
    dropout: float = 0.1  # Dropout 比例
    enc_in: int = 1  # 输入特征数目
    c_out: int = 1  # 输出特征数目
    embed: str = 'fixed'  # 嵌入类型
    freq: str = 'h'  # 时间序列的频率
    top_k: int = 5  # 高频成分数量
    e_layers: int = 2  # 网络层数
    num_kernels: int = 8
    label_len: int = 24

@dataclass
class timeLLMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timellm'
    task_name: str='timeLLM进行时序预测',
    bs: int = 1
    d_ff: int = 32
    top_k: int = 5
    llm_dim: int = 4096
    d_model: int = 16
    patch_len: int = 16
    stride: int = 8
    llm_model: str = "LLAMA"
    llm_layers: int =6
    prompt_domain: int = 0
    n_heads: int =8
    enc_in:int =7
    dropout: float = 0.1
    dataset: str = "weather"

@dataclass
class timeLLMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timellm'
    task_name: str='timeLLM进行时序预测',
    d_ff: int = 32
    top_k: int = 5
    llm_dim: int = 4096
    d_model: int = 16
    patch_len: int = 16
    stride: int = 8
    llm_model: str = "LLAMA"
    llm_layers: int =6
    prompt_domain: int = 0
    n_heads: int =8
    enc_in:int =7
    dropout: float = 0.1
    dataset: str = "weather"



@dataclass
class RNNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'rnn'
    bs: int = 128
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'
    bs: int = 128
    rank: int = 32
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