from configs.MainConfig import OtherConfig
from configs.default_config import *


@dataclass
class TimesNetConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'timesnet'  # 模型类型
    bs: int = 256  # 批大小
    epochs: int = 200  # 训练周期
    d_model: int = 512
   
    dropout: float = 0.1  # Dropout 比例
    enc_in: int = 1  # 输入特征数目
    c_out: int = 1  # 输出特征数目
    freq: str = 'h'  # 时间序列的频率
    top_k: int = 5  # 高频成分数量
    e_layers: int = 2  # 网络层数
    num_kernels: int = 6
    
    patience: int = 8
    verbose: int = 1
    num_layers: int = 3
    n_heads: int = 4
    revin: bool = True
    dropout: float = 0.1
    match_mode: str = 'abc'
    constraint: bool = False
    win_size: int = 48
    patch_len: int = 16
    d_ff: int = 512
    embed : str = 'timeF'
    factor: int = 3
    activation: str = 'gelu'
    freq: str = 'h'
    d_layers: int = 1
    task_name: str = 'long_term_forecast'  
   
    label_len: int = 48  
    dec_in : int = 1  
    distil: bool = True
    patch_size: int = 24
    
    
