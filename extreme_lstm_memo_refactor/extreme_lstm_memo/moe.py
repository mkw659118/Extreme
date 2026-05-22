from typing import Dict

import torch
import torch.nn as nn


class LSTMExpert(nn.Module):
    """
    Residual LSTM expert used inside the MoE backbone.
    """

    def __init__(self, d_model: int, expert_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=max(1, expert_layers),
            batch_first=True,
            dropout=dropout if expert_layers > 1 else 0.0,
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out, _ = self.lstm(x)
        return self.norm(out + residual)


class BackboneMoE(nn.Module):
    """
    Top-k Mixture-of-Experts forecasting backbone.

    Each expert processes the embedded sequence. Sparse top-k router weights are
    expanded to full expert weights, used to fuse expert sequences, and then
    projected to the final forecasting horizon.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        out_dim: int,
        num_experts: int = 3,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_layers: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.pred_len = pred_len
        self.out_dim = out_dim

        self.experts = nn.ModuleList(
            [
                LSTMExpert(d_model=d_model, expert_layers=expert_layers, dropout=dropout)
                for _ in range(num_experts)
            ]
        )
        self.fuse_norm = nn.RMSNorm(d_model)
        self.forecast_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * out_dim),
        )

    def _build_sparse_topk_weights(
        self,
        head_mix_weights: torch.Tensor,
        topk_experts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _ = head_mix_weights.shape
        full_weights = head_mix_weights.new_zeros((batch_size, self.num_experts))
        full_weights.scatter_(dim=1, index=topk_experts, src=head_mix_weights)
        return full_weights

    def forward(
        self,
        x_emb: torch.Tensor,
        head_mix_weights: torch.Tensor,
        topk_experts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_mix_weights = self._build_sparse_topk_weights(head_mix_weights, topk_experts)
        expert_outputs = torch.stack([expert(x_emb) for expert in self.experts], dim=1)

        mix = full_mix_weights.unsqueeze(-1).unsqueeze(-1)
        fused_seq = torch.sum(mix * expert_outputs, dim=1)
        fused_seq = self.fuse_norm(fused_seq)

        summary = torch.cat([fused_seq[:, -1, :], fused_seq.mean(dim=1)], dim=-1)
        point_pred = self.forecast_head(summary).view(x_emb.size(0), self.pred_len, self.out_dim)

        return {
            "mix_weights": full_mix_weights,
            "expert_sequences": expert_outputs,
            "fused_sequence": fused_seq,
            "point_pred": point_pred,
        }
