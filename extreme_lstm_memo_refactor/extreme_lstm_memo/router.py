import torch
import torch.nn as nn


class RouterFromEmbeddingPreTrain(nn.Module):
    """
    Router network that maps prior posterior probabilities q to expert logits.
    """

    def __init__(self, num_experts: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_experts, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, q_prior: torch.Tensor) -> torch.Tensor:
        return self.net(q_prior)
