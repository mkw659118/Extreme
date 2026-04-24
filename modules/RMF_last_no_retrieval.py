from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding


class StudentTMixturePrior(nn.Module):
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

    def reset_parameters(self):
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
            z = torch.cat([std, max_abs_dx, mean_abs_dx, last], dim=-1)
        else:
            z = torch.cat([std, max_abs_dx, mean_abs_dx], dim=-1)

        return z

    @staticmethod
    def _window_to_patches(x_used: torch.Tensor, patch_len: int) -> torch.Tensor:
        batch_size, length, channels = x_used.shape

        if patch_len <= 1:
            return x_used

        usable = (length // patch_len) * patch_len

        if usable == 0:
            return x_used.mean(dim=1, keepdim=True)

        x_trim = x_used[:, :usable, :].contiguous()
        x_patch = x_trim.view(
            batch_size,
            usable // patch_len,
            patch_len,
            channels,
        ).mean(dim=2)

        return x_patch

    def _component_log_prob(self, z: torch.Tensor) -> torch.Tensor:
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

        for patch_len in self.scales:
            x_scale = self._window_to_patches(x_used, patch_len=patch_len)
            z_scale = self.extract_state_vector(x_scale)

            z_scales.append(z_scale)

            log_comp_scale = self._component_log_prob(z_scale)
            log_comp_scales.append(log_comp_scale)

            q_scale = torch.softmax(
                (
                    log_comp_scale
                    + torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
                )
                / self.temperature,
                dim=-1,
            )
            q_scales.append(q_scale)

        if self.include_seq_level:
            x_seq = x_used.mean(dim=1, keepdim=True)
            z_seq = self.extract_state_vector(x_seq)

            z_scales.append(z_seq)

            log_comp_seq = self._component_log_prob(z_seq)
            log_comp_scales.append(log_comp_seq)

            q_seq = torch.softmax(
                (
                    log_comp_seq
                    + torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
                )
                / self.temperature,
                dim=-1,
            )
            q_scales.append(q_seq)

        alpha = torch.softmax(self.alpha_logits, dim=0)

        log_comp_stack = torch.stack(log_comp_scales, dim=1)
        fused_log_comp = torch.sum(
            log_comp_stack * alpha.view(1, -1, 1),
            dim=1,
        )

        log_pi = torch.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
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


class RouterFromEmbeddingPreTrain(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
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


class LSTMExpert(nn.Module):
    def __init__(
        self,
        d_model: int,
        expert_layers: int = 1,
        dropout: float = 0.1,
    ):
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
                LSTMExpert(
                    d_model=d_model,
                    expert_layers=expert_layers,
                    dropout=dropout,
                )
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

        full_weights = head_mix_weights.new_zeros(
            (batch_size, self.num_experts)
        )

        full_weights.scatter_(
            dim=1,
            index=topk_experts,
            src=head_mix_weights,
        )

        return full_weights

    def forward(
        self,
        x_emb: torch.Tensor,
        head_mix_weights: torch.Tensor,
        topk_experts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        full_mix_weights = self._build_sparse_topk_weights(
            head_mix_weights=head_mix_weights,
            topk_experts=topk_experts,
        )

        expert_outputs = torch.stack(
            [expert(x_emb) for expert in self.experts],
            dim=1,
        )

        mix = full_mix_weights.unsqueeze(-1).unsqueeze(-1)

        fused_seq = torch.sum(mix * expert_outputs, dim=1)
        fused_seq = self.fuse_norm(fused_seq)

        summary = torch.cat(
            [
                fused_seq[:, -1, :],
                fused_seq.mean(dim=1),
            ],
            dim=-1,
        )

        point_pred = self.forecast_head(summary).view(
            x_emb.size(0),
            self.pred_len,
            self.out_dim,
        )

        return {
            "mix_weights": full_mix_weights,
            "expert_sequences": expert_outputs,
            "fused_sequence": fused_seq,
            "point_pred": point_pred,
        }


class ExtremeLSTMMemo(nn.Module):
    """
    Ablation version: w/o Retrieval Bank.

    This version keeps:
        - Multi-scale Student-T state prior
        - Router from state posterior
        - Top-k LSTM-MoE backbone
        - Forecast head

    This version removes:
        - Retrieval memory bank
        - Key-value retrieval
        - Cosine nearest-neighbor search
        - RetrievalBetaGate
        - Heuristic retrieval fusion
        - Gate-based retrieval fusion
    """

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        e_layers: int,
        d_layers: int,
        dec_in: int = 3,
        out_dim: int = 1,
        config=None,
    ):
        super().__init__()

        self.config = config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.c_in = c_in
        self.dec_in = dec_in
        self.out_dim = out_dim

        self.dropout = self.config.dropout
        self.device = self.config.device

        self.num_experts = getattr(self.config, "num_experts", 4)
        self.top_k_experts = min(
            getattr(self.config, "top_k_experts", 2),
            self.num_experts,
        )

        # This ablation explicitly disables retrieval.
        self.use_retrieval = False

        self.state_balance_weight = float(
            getattr(self.config, "state_balance_weight", 0.02)
        )
        self.state_dom_cap = float(
            getattr(self.config, "state_dom_cap", 0.8)
        )

        include_last_value = bool(
            getattr(self.config, "pretrain_include_last", True)
        )
        state_dim = 4 if include_last_value else 3

        scales_cfg = getattr(
            self.config,
            "state_prior_scales",
            (1, 4, 8, 16),
        )

        if isinstance(scales_cfg, str):
            scales = tuple(
                int(s.strip())
                for s in scales_cfg.split(",")
                if s.strip()
            )
        else:
            scales = tuple(int(s) for s in scales_cfg)

        if len(scales) == 0:
            scales = (1,)

        self.state_prior = StudentTMixturePrior(
            num_components=self.num_experts,
            state_dim=state_dim,
            use_all_channels=bool(
                getattr(self.config, "state_prior_use_all_channels", True)
            ),
            include_last_value=include_last_value,
            scales=scales,
            include_seq_level=bool(
                getattr(self.config, "state_prior_include_seq_level", True)
            ),
            learnable_scale_weights=bool(
                getattr(self.config, "state_prior_learnable_scale_weights", True)
            ),
            min_scale=float(getattr(self.config, "pretrain_min_scale", 1e-4)),
            min_df=float(getattr(self.config, "pretrain_min_df", 2.1)),
            temperature=float(
                getattr(self.config, "state_prior_temperature", 1.0)
            ),
        )

        self.enc_embedding = DataEmbedding(
            c_in=c_in,
            d_model=d_model,
            dropout=self.dropout,
        )

        self.router = RouterFromEmbeddingPreTrain(
            num_experts=self.num_experts,
            hidden=getattr(self.config, "router_hidden", 64),
            dropout=self.dropout,
        )

        self.backbone = BackboneMoE(
            d_model=d_model,
            pred_len=pred_len,
            out_dim=out_dim,
            num_experts=self.num_experts,
            top_k=self.top_k_experts,
            dropout=min(self.dropout, 0.1),
            expert_layers=max(1, getattr(self.config, "expert_layers", 1)),
        )

        # Dummy parameter only prevents old gate-stage training loops from crashing.
        # It does not affect prediction, routing, or loss value.
        self.gate_stage_dummy = nn.Parameter(torch.zeros(1))

        self.register_buffer(
            "retrieval_gate_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

        self.latest_aux_dict = {}

    def get_state_prior_parameters(self):
        return self.state_prior.parameters()

    def pretrain_state_prior_loss(self, x: torch.Tensor):
        prior_out = self.state_prior(x)

        eps = 1e-8
        q = prior_out["q"]
        q_mean = q.mean(dim=0)

        uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())

        balance_kl = torch.sum(
            q_mean * (torch.log(q_mean + eps) - torch.log(uniform + eps))
        )

        dominant_penalty = F.relu(q_mean.max() - self.state_dom_cap).pow(2)

        loss = (
            prior_out["pretrain_nll"]
            + self.state_balance_weight * (balance_kl + dominant_penalty)
        )

        aux = {
            "pretrain_nll": prior_out["pretrain_nll"].detach(),
            "pretrain_total_loss": loss.detach(),
            "q_mean": q_mean.detach(),
            "mix_prob": prior_out["mix_prob"].detach(),
            "balance_kl": balance_kl.detach(),
            "dominant_penalty": dominant_penalty.detach(),
        }

        return loss, aux

    def freeze_state_prior(self):
        for p in self.state_prior.parameters():
            p.requires_grad = False

    def unfreeze_state_prior(self):
        for p in self.state_prior.parameters():
            p.requires_grad = True

    def freeze_backbone_for_gate(self):
        """
        In the full model, this stage trains the retrieval beta gate.

        In this ablation, retrieval is removed.
        Therefore, gate training should become a no-op.

        This method freezes all real model parameters and only leaves
        gate_stage_dummy trainable so an old training pipeline can still call
        backward() without changing the forecasting model.
        """
        for _, p in self.named_parameters():
            p.requires_grad = False

        self.gate_stage_dummy.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def mark_gate_ready(self, ready: bool = True):
        """
        Kept as a no-op for old experiment code.

        Retrieval is disabled in this ablation, so this flag has no effect.
        """
        self.retrieval_gate_ready.fill_(False)

    def construct_index(self, num: int):
        """
        No-op.

        Full model constructs a retrieval memory bank here.
        Ablation model does not build any retrieval index.
        """
        self.keys = None
        self.values = None
        self.index = 0

    @torch.no_grad()
    def add_key_value(
        self,
        x_enc: torch.Tensor,
        y: torch.Tensor,
        index: torch.Tensor,
    ):
        """
        No-op.

        Full model stores key-value samples for retrieval.
        Ablation model does not store anything.
        """
        return

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        raise RuntimeError(
            "Retrieval module is disabled in this ablation version."
        )

    def compute_sample_level_balance_loss(
        self,
        router_logits: torch.Tensor,
    ):
        router_prob = torch.softmax(router_logits, dim=-1)
        load = router_prob.mean(dim=0)

        min_load = 0.2
        max_load = 0.5

        low_penalty = F.relu(min_load - load).pow(2)
        high_penalty = F.relu(load - max_load).pow(2)

        balance_loss = (low_penalty + high_penalty).sum()

        aux_dict = {
            "balance_loss": balance_loss.detach(),
            "expert_load": load.detach(),
        }

        return balance_loss, aux_dict

    def _forward_backbone(self, x: torch.Tensor):
        x_emb = self.enc_embedding(x)

        prior_out = self.state_prior(x)

        router_logits = self.router(prior_out["q"])
        router_prob = torch.softmax(router_logits, dim=-1)

        topk_probs, topk_experts = torch.topk(
            router_prob,
            k=self.top_k_experts,
            dim=-1,
        )

        head_mix_weights = topk_probs / (
            topk_probs.sum(dim=-1, keepdim=True) + 1e-8
        )

        backbone_out = self.backbone(
            x_emb,
            head_mix_weights=head_mix_weights,
            topk_experts=topk_experts,
        )

        backbone_out.update(
            {
                "router_logits": router_logits,
                "router_prob": router_prob,
                "topk_experts": topk_experts,
                "topk_probs": head_mix_weights,
                "state_probs": prior_out["q"],
                "state_z": prior_out["z"],
                "state_alpha": prior_out["mix_prob"],
                "state_pretrain_nll": prior_out["pretrain_nll"],
            }
        )

        return backbone_out

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        mode: str = "train",
        return_aux: bool = False,
    ):
        """
        Forward path for w/o Retrieval Bank ablation.

        Output is always backbone point prediction:

            point_pred = MoE_Backbone(x)

        No retrieval prediction is computed.
        No retrieval fusion is applied.
        """

        if mode == "gate_train":
            with torch.no_grad():
                out = self._forward_backbone(x=x)
        else:
            out = self._forward_backbone(x=x)

        point_pred = out["point_pred"]

        total_aux_loss = point_pred.new_tensor(0.0)
        aux_dict = {}

        if mode in {"gate_train", "gate_valid"}:
            # Make old gate-stage training loops harmless.
            # This term has zero value and zero gradient effect on real modules.
            total_aux_loss = total_aux_loss + 0.0 * self.gate_stage_dummy.sum()

            aux_dict["beta_mean"] = point_pred.new_tensor(0.0).detach()
            aux_dict["beta_max"] = point_pred.new_tensor(0.0).detach()
            aux_dict["sim_mean"] = point_pred.new_tensor(0.0).detach()
            aux_dict["retrieval_disabled"] = point_pred.new_tensor(1.0).detach()

        if mode in {"train", "valid"}:
            balance_loss, balance_aux_dict = self.compute_sample_level_balance_loss(
                out["router_logits"]
            )

            total_aux_loss = total_aux_loss + 0.1 * balance_loss
            aux_dict.update(balance_aux_dict)

            eps = 1e-8
            q = out["state_probs"]
            q_mean = q.mean(dim=0)

            q_uniform = torch.full_like(q_mean, 1.0 / q_mean.numel())

            state_balance_loss = torch.sum(
                q_mean
                * (torch.log(q_mean + eps) - torch.log(q_uniform + eps))
            )

            state_dom_loss = F.relu(
                q_mean.max() - self.state_dom_cap
            ).pow(2)

            total_aux_loss = total_aux_loss + self.state_balance_weight * (
                state_balance_loss + state_dom_loss
            )

            aux_dict["state_balance_loss"] = state_balance_loss.detach()
            aux_dict["state_dom_loss"] = state_dom_loss.detach()
            aux_dict["state_qmax"] = q_mean.max().detach()

        out["point_pred"] = point_pred
        out["total_aux_loss"] = total_aux_loss

        aux_dict["total_aux_loss"] = total_aux_loss.detach()
        aux_dict["router_prob"] = out["router_prob"].detach()
        aux_dict["topk_experts"] = out["topk_experts"].detach()
        aux_dict["state_probs"] = out["state_probs"].detach()
        aux_dict["state_alpha"] = out["state_alpha"].detach()

        self.latest_aux_dict = aux_dict

        if return_aux:
            return out, total_aux_loss, aux_dict

        return out, total_aux_loss