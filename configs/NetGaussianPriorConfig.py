# coding: utf-8
from dataclasses import dataclass

from configs.NetConfig import NetConfig


@dataclass
class NetGaussianPriorConfig(NetConfig):
    state_prior_distribution: str = 'gaussian'
    experiment_tag: str = 'gaussian_prior'
