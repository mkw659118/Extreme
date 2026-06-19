# coding: utf-8
from dataclasses import dataclass

from configs.DATPNetMultiConfig import DATPNetMultiConfig


@dataclass
class DATPNetHorizonMultiConfig(DATPNetMultiConfig):
    model: str = 'datp_net_horizon_multi'
    experiment_tag: str = 'horizon_router'

    # Future-horizon Top-K MoE routing. The router returns [B, pred_len, E],
    # so each prediction horizon has its own expert distribution.
    router_granularity: str = 'horizon'
    horizon_router_use_state_context: bool = True

    # Keep all experts active while avoiding exactly uniform weights.
    router_balance_weight: float = 1.0
    topk_coverage_weight: float = 0.6
    router_entropy_weight: float = 0.004
    topk_min_usage: float = 0.12
    moe_weight_balance_weight: float = 2.0
    moe_min_weight: float = 0.08
    moe_max_weight: float = 0.55
    topk_weight_floor_weight: float = 1.0
    topk_min_head_weight: float = 0.18
    horizon_diversity_weight: float = 3.0
    horizon_min_weight_std: float = 0.04
    horizon_max_cosine: float = 0.75
    router_temperature: float = 0.9
    router_train_noise_std: float = 0.04
    ensure_all_experts_in_topk: bool = False

    # This diagnostic run focuses on the backbone MoE router.
    gate_epochs: int = 0
    pretrain_epochs: int = 0
