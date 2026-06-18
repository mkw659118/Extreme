# coding: utf-8
from dataclasses import dataclass

from configs.NetConfig import NetConfig


@dataclass
class DATPNetConfig(NetConfig):
    model: str = 'datp_net'
    num_experts: int = 4
    top_k_experts: int = 2
    retrieval_num: int = 2
    state_prior_scales: str = '1,4,8,16'
    state_prior_include_seq_level: bool = True
    use_retrieval: bool = True
    use_state_prior: bool = True
    use_missing_aware_encoding: bool = True

    # DATP-Net router coverage controls.
    # These make the trained Top-K routing easier to visualize and reduce
    # expert collapse in the hard Top-K activation counts.
    router_balance_weight: float = 0.1
    topk_coverage_weight: float = 0.25
    topk_min_usage: float = 0.12
    router_temperature: float = 1.0
    router_train_noise_std: float = 0.03
    ensure_all_experts_in_topk: bool = True
