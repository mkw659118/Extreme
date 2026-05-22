from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentTMixturePrior(nn.Module):
    """
    Multi-scale Student-t mixture prior for estimating latent temporal states.

    Given an input sequence x with shape [B, L, C], this module extracts compact
    state descriptors from multiple temporal scales and estimates a posterior
    distribution q over latent states. The posterior q is later used by the MoE
    router.
    """

    def __init__(
        self,
        num_components: int,
        state_dim: int,
        use_all_channels: bool = True,
        include_last_value: bool = True,
        scales: Sequence[int] = (1, 4, 8, 16),
        include_seq_level: bool = True,
        learnable_scale_weights: bool = True,
        min_scale: float = 1e-4,
        min_df: float = 2.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.state_dim = state_dim
        self.use_all_channels = use_all_channels
        self.include_last_value = include_last_value
        self.scales = tuple(scales)
        self.include_seq_level = include_seq_level
        self.min_scale = min_scale
        self.min_df = min_df
        self.temperature = temperature

        self.mu = nn.Parameter(torch.zeros(num_components, state_dim))
        self.scale_raw = nn.Parameter(torch.zeros(num_components, state_dim))
        self.df_raw = nn.Parameter(torch.zeros(num_components, state_dim))
        self.mix_logits = nn.Parameter(torch.zeros(num_components))
        self.register_buffer(
            "_log_pi",
            torch.log(torch.tensor(torch.pi, dtype=torch.float32)),
            persistent=False,
        )

        num_scale_terms = len(self.scales) + (1 if self.include_seq_level else 0)
        alpha_init = torch.zeros(num_scale_terms, dtype=torch.float32)
        if learnable_scale_weights:
            self.alpha_logits = nn.Parameter(alpha_init)
        else:
            self.register_buffer("alpha_logits", alpha_init, persistent=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize mixture parameters and break component symmetry."""
        with torch.no_grad():
            nn.init.normal_(self.mu, mean=0.0, std=0.05)
            if self.num_components > 1:
                comp_offsets = torch.linspace(
                    -0.5,
                    0.5,
                    steps=self.num_components,
                    device=self.mu.device,
                )
                self.mu[:, 0] = self.mu[:, 0] + comp_offsets

            nn.init.constant_(self.scale_raw, -1.0)
            nn.init.constant_(self.df_raw, 1.0)
            nn.init.normal_(self.mix_logits, mean=0.0, std=0.05)

    def extract_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [std, max_abs_dx, mean_abs_dx, optional_last_value]."""
        x_used = x if self.use_all_channels else x[:, :, :1]
        std = x_used.std(dim=1, unbiased=False).mean(dim=-1, keepdim=True)

        if x_used.size(1) > 1:
            dx = x_used[:, 1:, :] - x_used[:, :-1, :]
        else:
            dx = torch.zeros_like(x_used)

        max_abs_dx = dx.abs().amax(dim=1).mean(dim=-1, keepdim=True)
        mean_abs_dx = dx.abs().mean(dim=1).mean(dim=-1, keepdim=True)

        if self.include_last_value:
            last = x_used[:, -1, :].mean(dim=-1, keepdim=True)
            return torch.cat([std, max_abs_dx, mean_abs_dx, last], dim=-1)
        return torch.cat([std, max_abs_dx, mean_abs_dx], dim=-1)

    @staticmethod
    def _window_to_patches(x_used: torch.Tensor, patch_len: int) -> torch.Tensor:
        """Convert [B, L, C] to non-overlapping mean-pooled patches [B, N, C]."""
        batch_size, length, channels = x_used.shape
        if patch_len <= 1:
            return x_used

        usable = (length // patch_len) * patch_len
        if usable == 0:
            return x_used.mean(dim=1, keepdim=True)

        x_trim = x_used[:, :usable, :].contiguous()
        return x_trim.view(batch_size, usable // patch_len, patch_len, channels).mean(dim=2)

    def _component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Student-t component log probabilities with output shape [B, K]."""
        mu = self.mu.unsqueeze(0)
        scale = F.softplus(self.scale_raw).unsqueeze(0) + self.min_scale
        df = F.softplus(self.df_raw).unsqueeze(0) + self.min_df
        log_pi = self._log_pi.to(device=df.device, dtype=df.dtype)

        z_expand = z.unsqueeze(1)
        log_norm = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * (torch.log(df) + log_pi)
            - torch.log(scale)
        )
        log_kernel = -((df + 1.0) / 2.0) * torch.log1p(
            ((z_expand - mu) / scale).pow(2) / df
        )
        return (log_norm + log_kernel).sum(dim=-1)

    def posterior_from_z(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate posterior state probabilities from a pre-computed state vector."""
        log_comp = self._component_log_prob(z)
        log_pi = torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
        log_joint = log_comp + log_pi
        log_mix = torch.logsumexp(log_joint, dim=-1)
        q = torch.softmax(log_joint / self.temperature, dim=-1)
        return {
            "log_component": log_comp,
            "log_joint": log_joint,
            "log_mix": log_mix,
            "q": q,
            "mix_prob": torch.softmax(self.mix_logits, dim=0),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_used = x if self.use_all_channels else x[:, :, :1]

        z_scales = []
        log_comp_scales = []
        q_scales = []
        log_pi = torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)

        for patch_len in self.scales:
            x_scale = self._window_to_patches(x_used, patch_len=patch_len)
            z_scale = self.extract_state_vector(x_scale)
            z_scales.append(z_scale)

            log_comp_scale = self._component_log_prob(z_scale)
            log_comp_scales.append(log_comp_scale)
            q_scales.append(torch.softmax((log_comp_scale + log_pi) / self.temperature, dim=-1))

        if self.include_seq_level:
            x_seq = x_used.mean(dim=1, keepdim=True)
            z_seq = self.extract_state_vector(x_seq)
            z_scales.append(z_seq)

            log_comp_seq = self._component_log_prob(z_seq)
            log_comp_scales.append(log_comp_seq)
            q_scales.append(torch.softmax((log_comp_seq + log_pi) / self.temperature, dim=-1))

        alpha = torch.softmax(self.alpha_logits, dim=0)
        log_comp_stack = torch.stack(log_comp_scales, dim=1)
        fused_log_comp = torch.sum(log_comp_stack * alpha.view(1, -1, 1), dim=1)

        log_joint = fused_log_comp + log_pi
        log_mix = torch.logsumexp(log_joint, dim=-1)
        q = torch.softmax(log_joint / self.temperature, dim=-1)

        out = {
            "log_component": fused_log_comp,
            "log_joint": log_joint,
            "log_mix": log_mix,
            "q": q,
            "mix_prob": torch.softmax(self.mix_logits, dim=0),
            "z": torch.stack(z_scales, dim=1).mean(dim=1),
            "z_scales": torch.stack(z_scales, dim=1),
            "q_scales": torch.stack(q_scales, dim=1),
            "alpha": alpha,
        }
        out["pretrain_nll"] = -log_mix.mean()
        return out
