# coding: utf-8
from dataclasses import dataclass

from configs.ExtremeLSTMMemoPriorCompareConfig import ExtremeLSTMMemoPriorCompareConfig


@dataclass
class ExtremeLSTMMemoStudentTPriorConfig(ExtremeLSTMMemoPriorCompareConfig):
    state_prior_distribution: str = 'student_t'
    experiment_tag: str = 'student_t_prior'
