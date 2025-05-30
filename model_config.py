# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass

@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1

    seq_len: int = 24
    pred_len: int = 10
    ts_var: int = 0
    # start_date: str = '2022-07-13'
    # end_date: str = '2023-07-13'

    start_date: str = '2020-05-15'
    end_date: str = '2025-05-15'
    multi_dataset: bool = False

    seq_len: int = 96
    pred_len: int = 96
    ts_var: int = 0


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 56
    epochs: int = 200
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 45
    verbose: int = 5
    try_exp: int = 1
    dataset: str = 'financial'  # financial  weather
    multi_dataset: bool = True

    seq_len: int = 32
    pred_len: int = 10

    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'cosine'
    fft: bool = False
    revin: bool = False
    idx: int = 0


@dataclass
class TimeSeriesConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 56
    epochs: int = 200
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 45
    verbose: int = 5
    try_exp: int = 1
    dataset: str = 'weather'  # financial  weather
    multi_dataset: bool = False

    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'cosine'
    fft: bool = False
    revin: bool = False
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
    revin: bool = True


@dataclass
class MLP2Config(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp2'
    bs: int = 32
    rank: int = 32
    epochs: int = 200
    patience: int = 50
    verbose: int = 1
    num_layers: int = 2
    hidden_dim: int = 256
    revin: bool = True


@dataclass
class CrossformerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'crossformer'  
    bs: int = 256
    rank: int = 128
    verbose: int = 1
    seg_len: int = 24


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
class TimeLLMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
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
    llm_layers: int = 6
    prompt_domain: int = 0
    n_heads: int = 8
    enc_in:int = 7
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