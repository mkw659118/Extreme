# coding: utf-8
from dataclasses import dataclass

from configs.ExtremeLSTMMemoPriorCompareConfig import ExtremeLSTMMemoPriorCompareConfig


@dataclass
class ExtremeLSTMMemoGaussianPriorConfig(ExtremeLSTMMemoPriorCompareConfig):
    state_prior_distribution: str = 'gaussian'
    experiment_tag: str = 'gaussian_prior'
