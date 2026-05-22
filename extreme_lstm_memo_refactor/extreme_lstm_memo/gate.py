from typing import Optional, Tuple

import torch
import torch.nn as nn


class RetrievalBetaGate(nn.Module):
    """
    Adaptive gate for fusing parametric forecasts with retrieval-based forecasts.

    The gate produces beta in [beta_min, beta_max] with shape [B, 1, C], where
    larger beta means stronger reliance on the retrieved prediction.
    """

    def __init__(self, hidden_dim: int = 32, beta_min: float = 0.0, beta_max: float = 0.2):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mlp = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _reduce_sims(
        sims: Optional[torch.Tensor],
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sims is None:
            sim_mean = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            sim_max = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            sim_std = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            s = sims.unsqueeze(-1) if sims.dim() == 1 else sims.reshape(batch_size, -1)
            sim_mean = s.mean(dim=-1, keepdim=True)
            sim_max = s.max(dim=-1, keepdim=True).values
            sim_std = s.std(dim=-1, keepdim=True, unbiased=False)
        return (
            sim_mean.expand(batch_size, channels),
            sim_max.expand(batch_size, channels),
            sim_std.expand(batch_size, channels),
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        base_pred: torch.Tensor,
        ret_pred: torch.Tensor,
        sims: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, c_x = x_enc.shape
        _, _, c_y = base_pred.shape

        x_mean = x_enc.mean(dim=1)
        x_std = x_enc.std(dim=1, unbiased=False)
        x_last = x_enc[:, -1, :]
        if c_x != c_y:
            x_mean = x_mean[:, :c_y]
            x_std = x_std[:, :c_y]
            x_last = x_last[:, :c_y]

        p_mean = base_pred.mean(dim=1)
        p_std = base_pred.std(dim=1, unbiased=False)
        r_mean = ret_pred.mean(dim=1)
        r_std = ret_pred.std(dim=1, unbiased=False)
        diff_mean = (base_pred - ret_pred).abs().mean(dim=1)
        sim_mean, sim_max, sim_std = self._reduce_sims(
            sims,
            batch_size,
            c_y,
            x_enc.device,
            x_enc.dtype,
        )

        feat = torch.stack(
            [
                x_mean,
                x_std,
                x_last,
                p_mean,
                p_std,
                r_mean,
                r_std,
                diff_mean,
                sim_mean,
                sim_max,
                sim_std,
            ],
            dim=-1,
        )

        beta = torch.sigmoid(self.mlp(feat)).transpose(1, 2)
        return self.beta_min + (self.beta_max - self.beta_min) * beta
