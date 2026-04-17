#Author  :   mkw 
#Time    :   2025/09/18 20:07:57
#Desc    :   None


from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class MCANNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mcann'
    bs: int = 256
    epochs: int = 200
    gpu_id: int = 0
    atten_dim: int = 384
    d_model: int = 256
    input_dim: int = 1
    seq_len: int = 360
    layer: int = 4
    learning_rate: float = 0.001
    lradj: str = "type4"
    mode: str = "train"
    name: str = "Almaden"
    ngpu: int = 1
    os_s: int = 18
    os_v: int = 4
    outf: str = "./output"
    output_dim: int = 1
    pred_len: int = 8
    oversampling: int = 40
    rain_sensor: str = "reservoir_stor_4001_sof24"
    save: int = 0
    seq_weight: int = 0
    start_point: str = "1991-07-01 23:30:00"
    reservoir_sensor: str = "reservoir_stor_4001_sof24"
    test_end: str = "2019-07-01 00:30:00"
    test_start: str = "2018-07-01 00:30:00"
    train_point: str = "2018-06-30 23:30:00"
    train_seed: int = 2024
    train_volume: int = 40000
    val_seed: int = 2025
    val_size: int = 60
    watershed: int = 0
    patience: int = 40


