from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding import DataEmbedding

from .gate import RetrievalBetaGate
from .losses import compute_sample_level_balance_loss, compute_state_balance_loss
from .moe import BackboneMoE
from .prior import StudentTMixturePrior
from .retrieval import RetrievalMemory
from .router import RouterFromEmbeddingPreTrain


class ExtremeLSTMMemo(nn.Module):
    """
    Extreme-aware LSTM-MoE forecasting model with state prior and retrieval correction.

    The public interface is kept compatible with the original implementation:
    - construct_index(num)
    - add_key_value(x_enc, y, index)
    - retrieval(x, index)
    - forward(x, x_mark=None, dec_input=None, sample_ids=None, mode='train', return_aux=False)
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
        if config is None:
            raise ValueError("config must be provided because dropout/device and method hyperparameters are read from it.")

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
        self.top_k_experts = min(getattr(self.config, "top_k_experts", 2), self.num_experts)
        self.retrieval_num = getattr(self.config, "retrieval_num", 2)
        self.retrieval_stride = 1

        self.retrieval_tau = getattr(self.config, "retrieval_tau", 0.55)
        self.retrieval_alpha_max = getattr(self.config, "retrieval_alpha_max", 0.02)
        self.retrieval_beta_hidden = getattr(self.config, "retrieval_beta_hidden", 32)
        self.retrieval_beta_max = getattr(self.config, "retrieval_beta_max", 0.20)
        self.retrieval_beta_reg = getattr(self.config, "retrieval_beta_reg", 1e-4)
        self.state_balance_weight = float(getattr(self.config, "state_balance_weight", 0.02))
        self.state_dom_cap = float(getattr(self.config, "state_dom_cap", 0.8))

        include_last_value = bool(getattr(self.config, "pretrain_include_last", True))
        state_dim = 4 if include_last_value else 3
        scales = self._parse_scales(getattr(self.config, "state_prior_scales", (1, 4, 8, 16)))

        self.state_prior = StudentTMixturePrior(
            num_components=self.num_experts,
            state_dim=state_dim,
            use_all_channels=bool(getattr(self.config, "state_prior_use_all_channels", True)),
            include_last_value=include_last_value,
            scales=scales,
            include_seq_level=bool(getattr(self.config, "state_prior_include_seq_level", True)),
            learnable_scale_weights=bool(getattr(self.config, "state_prior_learnable_scale_weights", True)),
            min_scale=float(getattr(self.config, "pretrain_min_scale", 1e-4)),
            min_df=float(getattr(self.config, "pretrain_min_df", 2.1)),
            temperature=float(getattr(self.config, "state_prior_temperature", 1.0)),
        )

        self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
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
        self.beta_gate = RetrievalBetaGate(
            hidden_dim=self.retrieval_beta_hidden,
            beta_min=0.0,
            beta_max=self.retrieval_beta_max,
        )

        self.memory = RetrievalMemory(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            c_in=self.c_in,
            value_dim=self.dec_in,
            device=self.device,
            stride=self.retrieval_stride,
        )

        self.register_buffer("retrieval_gate_ready", torch.tensor(False, dtype=torch.bool), persistent=True)
        self.latest_aux_dict: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _parse_scales(scales_cfg) -> Tuple[int, ...]:
        if isinstance(scales_cfg, str):
            scales = tuple(int(s.strip()) for s in scales_cfg.split(",") if s.strip())
        else:
            scales = tuple(int(s) for s in scales_cfg)
        return scales if len(scales) > 0 else (1,)

    def get_state_prior_parameters(self):
        return self.state_prior.parameters()

    def pretrain_state_prior_loss(self, x: torch.Tensor):
        prior_out = self.state_prior(x)
        state_balance_loss, state_dom_loss, q_mean = compute_state_balance_loss(
            prior_out["q"],
            state_dom_cap=self.state_dom_cap,
        )
        loss = prior_out["pretrain_nll"] + self.state_balance_weight * (
            state_balance_loss + state_dom_loss
        )
        aux = {
            "pretrain_nll": prior_out["pretrain_nll"].detach(),
            "pretrain_total_loss": loss.detach(),
            "q_mean": q_mean.detach(),
            "mix_prob": prior_out["mix_prob"].detach(),
            "balance_kl": state_balance_loss.detach(),
            "dominant_penalty": state_dom_loss.detach(),
        }
        return loss, aux

    def freeze_state_prior(self) -> None:
        for p in self.state_prior.parameters():
            p.requires_grad = False

    def unfreeze_state_prior(self) -> None:
        for p in self.state_prior.parameters():
            p.requires_grad = True

    def freeze_backbone_for_gate(self) -> None:
        for _, p in self.named_parameters():
            p.requires_grad = False
        for p in self.beta_gate.parameters():
            p.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def mark_gate_ready(self, ready: bool = True) -> None:
        self.retrieval_gate_ready.fill_(bool(ready))

    def compute_sample_level_balance_loss(self, router_logits: torch.Tensor):
        return compute_sample_level_balance_loss(router_logits)

    def construct_index(self, num: int) -> None:
        self.memory.construct_index(num)

    @torch.no_grad()
    def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> None:
        self.memory.add_key_value(x_enc, y, index)

    @property
    def index(self) -> int:
        return self.memory.index

    @property
    def keys(self) -> Optional[torch.Tensor]:
        return self.memory.keys

    @property
    def values(self) -> Optional[torch.Tensor]:
        return self.memory.values

    def cosine_similarity(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return self.memory.cosine_similarity(queries, keys)

    def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
        return self.memory.query(
            x=x,
            sample_ids=index,
            top_k=self.retrieval_num,
            exclude_self_window=self.training,
        )

    def _forward_backbone(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_emb = self.enc_embedding(x)
        prior_out = self.state_prior(x)
        router_logits = self.router(prior_out["q"])
        router_prob = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
        head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

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

    def _heuristic_fuse(
        self,
        point_pred: torch.Tensor,
        retrieval_pred: torch.Tensor,
        sims: torch.Tensor,
    ):
        sim_mean = sims.mean(dim=-1)
        dynamic_alpha = (sim_mean - self.retrieval_tau) / (1.0 - self.retrieval_tau + 1e-8)
        dynamic_alpha = dynamic_alpha.clamp(0.0, 1.0)
        dynamic_alpha = self.retrieval_alpha_max * dynamic_alpha
        dynamic_alpha = dynamic_alpha.view(-1, 1, 1)
        fused = (1 - dynamic_alpha) * point_pred + dynamic_alpha * retrieval_pred
        return fused, dynamic_alpha

    def _gate_fuse(
        self,
        x: torch.Tensor,
        point_pred: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
    ):
        retrieval_results, sims, _ = self.retrieval(x, sample_ids)
        retrieval_pred = retrieval_results[:, :, : self.out_dim]
        beta = self.beta_gate(
            x_enc=x,
            base_pred=point_pred.detach(),
            ret_pred=retrieval_pred.detach(),
            sims=sims,
        )
        fused_point = (1.0 - beta) * point_pred + beta * retrieval_pred
        return fused_point, retrieval_pred, sims, beta

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        dec_input: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        mode: str = "train",
        return_aux: bool = False,
    ):
        del x_mark, dec_input

        if mode == "gate_train":
            with torch.no_grad():
                out = self._forward_backbone(x=x)
        else:
            out = self._forward_backbone(x=x)

        point_pred = out["point_pred"]
        total_aux_loss = point_pred.new_tensor(0.0)
        aux_dict: Dict[str, torch.Tensor] = {}

        if mode in {"gate_train", "gate_valid"}:
            fused_point, retrieval_pred, sims, beta = self._gate_fuse(x, point_pred, sample_ids)
            point_pred = fused_point
            total_aux_loss = total_aux_loss + self.retrieval_beta_reg * beta.mean()
            aux_dict["beta_mean"] = beta.mean().detach()
            aux_dict["beta_max"] = beta.max().detach()
            aux_dict["sim_mean"] = sims.mean().detach()
            out["retrieval_pred"] = retrieval_pred
            out["beta"] = beta
        elif mode == "test" and self.index > 0:
            if bool(self.retrieval_gate_ready.item()):
                fused_point, retrieval_pred, sims, beta = self._gate_fuse(x, point_pred, sample_ids)
                point_pred = fused_point
                aux_dict["beta_mean"] = beta.mean().detach()
                aux_dict["beta_max"] = beta.max().detach()
                aux_dict["sim_mean"] = sims.mean().detach()
                out["retrieval_pred"] = retrieval_pred
                out["beta"] = beta
            else:
                retrieval_results, sims, _ = self.retrieval(x, sample_ids)
                retrieval_pred = retrieval_results[:, :, : self.out_dim]
                fused_point, dynamic_alpha = self._heuristic_fuse(point_pred, retrieval_pred, sims)
                point_pred = fused_point
                aux_dict["beta_mean"] = dynamic_alpha.mean().detach()
                aux_dict["beta_max"] = dynamic_alpha.max().detach()
                aux_dict["sim_mean"] = sims.mean().detach()
                out["retrieval_pred"] = retrieval_pred
                out["beta"] = dynamic_alpha

        if mode in {"train", "valid"}:
            balance_loss, balance_aux_dict = self.compute_sample_level_balance_loss(out["router_logits"])
            total_aux_loss = total_aux_loss + 0.1 * balance_loss
            aux_dict.update(balance_aux_dict)

            state_balance_loss, state_dom_loss, q_mean = compute_state_balance_loss(
                out["state_probs"],
                state_dom_cap=self.state_dom_cap,
            )
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
