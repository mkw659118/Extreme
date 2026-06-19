# coding: utf-8
from dataclasses import dataclass

from configs.DATPNetMultiConfig import DATPNetMultiConfig


@dataclass
class DATPNetStepMultiConfig(DATPNetMultiConfig):
    model: str = 'datp_net_step_multi'
    experiment_tag: str = 'step_router'

    # Step-level Top-K MoE routing.
    router_granularity: str = 'step'
    step_router_use_state_context: bool = True

    # Balance all experts over all batch-time tokens, and directly regularize
    # the sparse MoE weights used by the backbone. This avoids "selected but
    # near-zero weight" false activations in Top-K visualizations.
    router_balance_weight: float = 1.0
    topk_coverage_weight: float = 0.6
    router_entropy_weight: float = 0.004
    topk_min_usage: float = 0.12
    moe_weight_balance_weight: float = 2.0
    moe_min_weight: float = 0.08
    moe_max_weight: float = 0.55
    topk_weight_floor_weight: float = 1.0
    topk_min_head_weight: float = 0.18
    router_temperature: float = 0.9
    router_train_noise_std: float = 0.04
    ensure_all_experts_in_topk: bool = False

    # The MoE visualization only needs the backbone router; skip retrieval-gate
    # fine-tuning so this diagnostic run is faster and easier to interpret.
    gate_epochs: int = 0
    pretrain_epochs: int = 0
