from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_sample_level_balance_loss(
    router_logits: torch.Tensor,
    min_load: float = 0.2,
    max_load: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize overly imbalanced expert usage at the batch level."""
    router_prob = torch.softmax(router_logits, dim=-1)
    load = router_prob.mean(dim=0)
    low_penalty = F.relu(min_load - load).pow(2)
    high_penalty = F.relu(load - max_load).pow(2)
    balance_loss = (low_penalty + high_penalty).sum()
    aux_dict = {
        "balance_loss": balance_loss.detach(),
        "expert_load": load.detach(),
    }
    return balance_loss, aux_dict


def compute_state_balance_loss(
    state_probs: torch.Tensor,
    state_dom_cap: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep state-prior posterior assignments from collapsing to one component."""
    q_mean = state_probs.mean(dim=0)
    q_uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())
    state_balance_loss = torch.sum(q_mean * (torch.log(q_mean + eps) - torch.log(q_uniform + eps)))
    state_dom_loss = F.relu(q_mean.max() - state_dom_cap).pow(2)
    return state_balance_loss, state_dom_loss, q_mean
