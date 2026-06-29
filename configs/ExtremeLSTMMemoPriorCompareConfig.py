# coding: utf-8
from dataclasses import dataclass

from configs.ExtremeLSTMMemoConfig import ExtremeLSTMMemoConfig


@dataclass
class ExtremeLSTMMemoPriorCompareConfig(ExtremeLSTMMemoConfig):
    model: str = 'extreme_lstm_memo_prior_compare'
    num_experts: int = 4
    top_k_experts: int = 2
    retrieval_num: int = 2
    pretrain_epochs: int = 10
    state_prior_distribution: str = 'student_t'
    state_prior_scales: str = '1,4,8,16'
    state_prior_include_seq_level: bool = True
    experiment_tag: str = 'prior_compare'
