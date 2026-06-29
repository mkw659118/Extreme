# coding: utf-8
from dataclasses import dataclass

from configs.NetConfig import NetConfig


@dataclass
class NetStudentTPriorConfig(NetConfig):
    state_prior_distribution: str = 'student_t'
    experiment_tag: str = 'student_t_prior'
