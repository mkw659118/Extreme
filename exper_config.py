# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 50
    device: str = 'cuda'
    epochs: int = 200
    patience: int = 50
    verbose: int = 10
    try_exp: int = 1


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    bs: int = 32
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class RNNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'rnn'
    bs: int = 128
    rank: int = 50
    epochs: int = 200
    patience: int = 50
    verbose: int = 1

@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'
    bs: int = 128
    rank: int = 50
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

@dataclass
class CrossformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'crossformer'  
    bs: int = 256
    rank: int = 50  
    epochs: int = 200  #
    patience: int = 50  
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
