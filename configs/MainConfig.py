# coding : utf-8
# Author : yuxiang Zeng

from dataclasses import dataclass
@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1

    ts_var: int = 1
    # start_date: str = '2022-07-13'
    # end_date: str = '2023-07-13'
    start_date: str = '2020-05-15'
    end_date: str = '2025-05-15'
    multi_dataset: bool = False
    seq_len: int = 96
    pred_len: int = 96














