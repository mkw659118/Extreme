from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.att.cross_attention import CrossAttention
from layers.embedding import DataEmbedding


class HomogeneousPointExpert(nn.Module):
	def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.1):
		super().__init__()
		# Use a narrower bottleneck by default to keep the expert lightweight.
		hidden = hidden or max(d_model // 2, 16)
		self.fc1 = nn.Linear(d_model, hidden)
		self.fc2 = nn.Linear(hidden, d_model)
		self.act = nn.GELU()
		self.drop = nn.Dropout(dropout)
		self.norm = nn.RMSNorm(d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = x
		x = self.drop(self.act(self.fc1(x)))
		x = self.fc2(x)
		return self.norm(residual + x)


class MultiScaleStudentTStatePrior(nn.Module):
	def __init__(
		self,
		use_all_channels: bool = False,
		scales: Sequence[int] = (1, 3, 6, 12, 24),
		temperature: float = 1.25,
		learnable_prototypes: bool = False,
		learnable_scale_weights: bool = True,
		normal_scale: float = 0.45,
		mid_scale: float = 1.40,
		extreme_scale: float = 4.20,
		normal_df: float = 40.0,
		mid_df: float = 7.0,
		extreme_df: float = 2.15,
		min_scale: float = 1e-4,
		min_df: float = 2.1,
		lambda_mean: float = 0.25,
		lambda_max: float = 0.60,
		lambda_last: float = 0.15,
	):
		super().__init__()
		self.use_all_channels = use_all_channels
		self.scales = tuple(scales)
		self.temperature = temperature
		self.min_scale = min_scale
		self.min_df = min_df

		# Keep aggregation weights numerically sane even if user passes unusual values.
		lam_sum = lambda_mean + lambda_max + lambda_last
		lam_sum = lam_sum if lam_sum > 1e-8 else 1.0
		self.lambda_mean = lambda_mean / lam_sum
		self.lambda_max = lambda_max / lam_sum
		self.lambda_last = lambda_last / lam_sum

		init_mu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
		init_scale = torch.tensor([normal_scale, mid_scale, extreme_scale], dtype=torch.float32)
		init_df = torch.tensor([normal_df, mid_df, extreme_df], dtype=torch.float32)

		if learnable_prototypes:
			self.proto_mu = nn.Parameter(init_mu)
			self.proto_scale_raw = nn.Parameter(torch.log(torch.exp(init_scale - min_scale) - 1.0))
			self.proto_df_raw = nn.Parameter(torch.log(torch.exp(init_df - min_df) - 1.0))
		else:
			self.register_buffer('proto_mu', init_mu, persistent=True)
			self.register_buffer('proto_scale', init_scale, persistent=True)
			self.register_buffer('proto_df', init_df, persistent=True)
			self.proto_scale_raw = None
			self.proto_df_raw = None

		self.scale_names = [f'patch{m}' for m in self.scales] + ['seq']
		alpha_init = torch.zeros(len(self.scale_names), dtype=torch.float32)
		if learnable_scale_weights:
			self.alpha_logits = nn.Parameter(alpha_init)
		else:
			self.register_buffer('alpha_logits', alpha_init, persistent=True)

		self.register_buffer('_log_pi', torch.log(torch.tensor(torch.pi, dtype=torch.float32)), persistent=False)

	def _get_proto_params(self):
		if self.proto_scale_raw is None:
			return self.proto_mu, self.proto_scale, self.proto_df
		mu = self.proto_mu
		scale = F.softplus(self.proto_scale_raw) + self.min_scale
		df = F.softplus(self.proto_df_raw) + self.min_df
		return mu, scale, df

	def _student_t_log_prob(self, x: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, df: torch.Tensor):
		x = x.unsqueeze(2)  # [B, Npatch, 1, Nelem]
		mu = mu.view(1, 1, 3, 1)
		scale = scale.view(1, 1, 3, 1)
		df = df.view(1, 1, 3, 1)

		z = (x - mu) / scale
		log_pi = self._log_pi.to(device=x.device, dtype=x.dtype)
		log_norm = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0) - 0.5 * (torch.log(df) + log_pi) - torch.log(scale)
		log_kernel = -((df + 1.0) / 2.0) * torch.log1p((z ** 2) / df)
		return log_norm + log_kernel

	@staticmethod
	def _window_to_patches(x_used: torch.Tensor, patch_len: int) -> torch.Tensor:
		bsz, length, channels = x_used.shape
		if patch_len <= 1:
			return x_used.reshape(bsz, length, channels)
		usable = (length // patch_len) * patch_len
		if usable == 0:
			return x_used.reshape(bsz, 1, length * channels)
		x_trim = x_used[:, :usable, :].contiguous()
		return x_trim.reshape(bsz, usable // patch_len, patch_len * channels)

	def _aggregate_patch_scores(self, patch_scores: torch.Tensor) -> torch.Tensor:
		mean_score = patch_scores.mean(dim=1)
		max_score = patch_scores.max(dim=1).values
		last_score = patch_scores[:, -1, :]
		return self.lambda_mean * mean_score + self.lambda_max * max_score + self.lambda_last * last_score

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		x_used = x if self.use_all_channels else x[:, :, :1]
		bsz = x_used.size(0)
		mu, scale, df = self._get_proto_params()

		scale_scores = []
		for m in self.scales:
			patches = self._window_to_patches(x_used, patch_len=m)
			log_prob = self._student_t_log_prob(patches, mu, scale, df)
			patch_scores = log_prob.mean(dim=-1)
			scale_scores.append(self._aggregate_patch_scores(patch_scores))

		seq_patch = x_used.reshape(bsz, 1, -1)
		seq_score = self._student_t_log_prob(seq_patch, mu, scale, df).mean(dim=-1).squeeze(1)
		scale_scores.append(seq_score)

		scale_scores = torch.stack(scale_scores, dim=1)  # [B, S, 3]
		alpha = torch.softmax(self.alpha_logits, dim=0)
		fused_scores = torch.sum(scale_scores * alpha.view(1, -1, 1), dim=1)

		return {
			'q_final': torch.softmax(fused_scores / self.temperature, dim=-1),
			'fused_scores': fused_scores,
			'scale_scores': scale_scores,
			'q_scales': torch.softmax(scale_scores / self.temperature, dim=-1),
			'alpha': alpha,
		}


class StepRouterFromX(nn.Module):
	def __init__(
		self,
		c_in: int,
		d_model: int,
		num_experts: int,
		hidden: int = 32,
		dropout: float = 0.0,
		state_prior_use_all_channels: bool = False,
		state_prior_temperature: float = 1.25,
		state_prior_learnable: bool = False,
		state_prior_scales: Sequence[int] = (1, 3, 6, 12, 24),
		state_prior_scale_weights_learnable: bool = True,
	):
		super().__init__()
		self.state_prior = MultiScaleStudentTStatePrior(
			use_all_channels=state_prior_use_all_channels,
			scales=state_prior_scales,
			temperature=state_prior_temperature,
			learnable_prototypes=state_prior_learnable,
			learnable_scale_weights=state_prior_scale_weights_learnable,
		)

		in_dim = 3 * c_in + 3
		self.global_net = nn.Sequential(
			nn.Linear(in_dim, hidden),
			nn.LayerNorm(hidden),
			nn.GELU(),
			nn.Dropout(dropout),
		)
		self.step_proj = nn.Linear(d_model, hidden)
		self.out_net = nn.Sequential(
			nn.LayerNorm(hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, num_experts),
		)

	def forward(self, x: torch.Tensor, step_feat: torch.Tensor):
		prior_out = self.state_prior(x)
		global_feat = torch.cat([
			x.std(dim=1, unbiased=False),
			x.abs().amax(dim=1),
			x[:, -1, :],
			prior_out['q_final'],
		], dim=-1)
		global_h = self.global_net(global_feat)
		step_h = self.step_proj(step_feat)
		logits = self.out_net(step_h + global_h.unsqueeze(1))
		return logits, prior_out


class PointMoEHead(nn.Module):
	def __init__(
		self,
		d_model: int,
		out_dim: int = 1,
		num_experts: int = 3,
		top_k: int = 2,
		dropout: float = 0.1,
		expert_hidden: Optional[int] = None,
	):
		super().__init__()
		self.num_experts = num_experts
		self.top_k = min(top_k, num_experts)
		hidden = expert_hidden or max(d_model // 2, 16)
		self.experts = nn.ModuleList([HomogeneousPointExpert(d_model=d_model, hidden=hidden, dropout=dropout) for _ in range(num_experts)])
		self.point_heads = nn.ModuleList([nn.Linear(d_model, out_dim) for _ in range(num_experts)])

	def _build_sparse_topk_weights(self, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> torch.Tensor:
		bsz, horizon, _ = head_mix_weights.shape
		full_weights = head_mix_weights.new_zeros((bsz, horizon, self.num_experts))
		full_weights.scatter_(dim=2, index=topk_experts, src=head_mix_weights)
		return full_weights

	def forward(self, x: torch.Tensor, head_mix_weights: torch.Tensor, topk_experts: torch.Tensor) -> Dict[str, torch.Tensor]:
		full_mix_weights = self._build_sparse_topk_weights(head_mix_weights, topk_experts)
		expert_points = torch.stack([self.point_heads[e](self.experts[e](x)) for e in range(self.num_experts)], dim=1)
		mix = full_mix_weights.permute(0, 2, 1).unsqueeze(-1)
		point_pred = torch.sum(mix * expert_points, dim=1)
		return {
			'mix_weights': full_mix_weights,
			'expert_points': expert_points,
			'point_pred': point_pred,
		}


class RetrievalBetaGate(nn.Module):
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
	def _reduce_sims(sims: Optional[torch.Tensor], bsz: int, channels: int, device: torch.device, dtype: torch.dtype):
		if sims is None:
			sim_mean = torch.zeros(bsz, 1, device=device, dtype=dtype)
			sim_max = torch.zeros(bsz, 1, device=device, dtype=dtype)
			sim_std = torch.zeros(bsz, 1, device=device, dtype=dtype)
		else:
			s = sims.unsqueeze(-1) if sims.dim() == 1 else sims.reshape(bsz, -1)
			sim_mean = s.mean(dim=-1, keepdim=True)
			sim_max = s.max(dim=-1, keepdim=True).values
			sim_std = s.std(dim=-1, keepdim=True, unbiased=False)
		return sim_mean.expand(bsz, channels), sim_max.expand(bsz, channels), sim_std.expand(bsz, channels)

	def forward(self, x_enc: torch.Tensor, base_pred: torch.Tensor, ret_pred: torch.Tensor, sims: Optional[torch.Tensor]) -> torch.Tensor:
		bsz, _, c_x = x_enc.shape
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
		sim_mean, sim_max, sim_std = self._reduce_sims(sims, bsz, c_y, x_enc.device, x_enc.dtype)

		feat = torch.stack([
			x_mean, x_std, x_last,
			p_mean, p_std,
			r_mean, r_std,
			diff_mean,
			sim_mean, sim_max, sim_std,
		], dim=-1)

		beta = torch.sigmoid(self.mlp(feat)).transpose(1, 2)
		return self.beta_min + (self.beta_max - self.beta_min) * beta


class ExtremeLSTMMemo(nn.Module):
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

		self.num_experts = getattr(self.config, 'num_experts', 3)
		self.retrieval_num = getattr(self.config, 'retrieval_num', 2)
		self.top_k_experts = min(getattr(self.config, 'top_k_experts', 2), self.num_experts)
		self.retrieval_stride = 1

		self.retrieval_tau = getattr(self.config, 'retrieval_tau', 0.55)
		self.retrieval_alpha_max = getattr(self.config, 'retrieval_alpha_max', 0.02)
		self.retrieval_beta_hidden = getattr(self.config, 'retrieval_beta_hidden', 32)
		self.retrieval_beta_max = getattr(self.config, 'retrieval_beta_max', 0.20)
		self.retrieval_beta_reg = getattr(self.config, 'retrieval_beta_reg', 1e-4)
		self.state_prior_balance_reg = getattr(self.config, 'state_prior_balance_reg', 1e-3)

		self.register_buffer('retrieval_gate_ready', torch.tensor(False, dtype=torch.bool), persistent=True)

		self.enc_embedding = DataEmbedding(c_in=c_in, d_model=d_model, dropout=self.dropout)
		self.dec_embedding = DataEmbedding(c_in=dec_in, d_model=d_model, dropout=self.dropout)

		self.encoder = nn.LSTM(
			input_size=d_model,
			hidden_size=d_model,
			num_layers=e_layers,
			batch_first=True,
			dropout=self.dropout if e_layers > 1 else 0.0,
		)
		self.decoder = nn.LSTM(
			input_size=d_model,
			hidden_size=d_model,
			num_layers=d_layers,
			batch_first=True,
			dropout=self.dropout if d_layers > 1 else 0.0,
		)

		self.post_norm = nn.RMSNorm(d_model)
		self.xattn = CrossAttention(d_model)
		self.fuse_proj = nn.Linear(2 * d_model, d_model)

		self.router = StepRouterFromX(
			c_in=c_in,
			d_model=d_model,
			num_experts=self.num_experts,
			hidden=getattr(self.config, 'router_hidden', 32),
			dropout=self.dropout,
			state_prior_use_all_channels=getattr(self.config, 'state_prior_use_all_channels', True),
			state_prior_temperature=getattr(self.config, 'state_prior_temperature', 1.25),
			state_prior_learnable=getattr(self.config, 'state_prior_learnable', True),
			state_prior_scales=getattr(self.config, 'state_prior_scales', (1, 3, 6, 12, 24)),
			state_prior_scale_weights_learnable=getattr(self.config, 'state_prior_scale_weights_learnable', True),
		)

		self.moe_head = PointMoEHead(
			d_model=d_model,
			out_dim=out_dim,
			num_experts=self.num_experts,
			top_k=self.top_k_experts,
			dropout=min(self.dropout, 0.1),
			expert_hidden=getattr(self.config, 'expert_hidden', None),
		)

		self.beta_gate = RetrievalBetaGate(
			hidden_dim=self.retrieval_beta_hidden,
			beta_min=0.0,
			beta_max=self.retrieval_beta_max,
		)

		self.latest_aux_dict = {}

	def freeze_backbone_for_gate(self):
		for _, p in self.named_parameters():
			p.requires_grad = False
		for p in self.beta_gate.parameters():
			p.requires_grad = True

	def unfreeze_all(self):
		for p in self.parameters():
			p.requires_grad = True

	def mark_gate_ready(self, ready: bool = True):
		self.retrieval_gate_ready.fill_(bool(ready))

	def compute_sample_level_balance_loss(self, router_logits: torch.Tensor):
		"""
		Load balancing for MoE routing.
		
		Compared with fixed min/max thresholds, this objective is adaptive to the
		number of experts and better aligned with top-k routing behavior.
		"""
		router_prob = torch.softmax(router_logits, dim=-1)
		if router_prob.dim() == 3:
			# [B, L, E] -> [T, E]
			flat_prob = router_prob.reshape(-1, router_prob.size(-1))
		else:
			flat_prob = router_prob

		num_experts = flat_prob.size(-1)
		target = flat_prob.new_full((num_experts,), 1.0 / float(num_experts))

		# Importance: expected probability mass each expert receives.
		importance = flat_prob.mean(dim=0)

		# Top-k dispatched mass: closer to the actual sparse routing usage.
		topk = min(self.top_k_experts, num_experts)
		topk_idx = torch.topk(flat_prob, k=topk, dim=-1).indices
		mask = torch.zeros_like(flat_prob)
		mask.scatter_(dim=-1, index=topk_idx, value=1.0)
		dispatched_mass = flat_prob * mask
		load = dispatched_mass.sum(dim=0)
		load = load / (load.sum() + 1e-8)

		# Two complementary terms:
		# 1) importance balance (soft global balance)
		# 2) dispatched-load balance (top-k-aligned balance)
		importance_loss = F.mse_loss(importance, target)
		load_loss = F.mse_loss(load, target)
		balance_loss = 0.5 * importance_loss + 0.5 * load_loss

		aux_dict = {
			'balance_loss': balance_loss.detach(),
			'importance_loss': importance_loss.detach(),
			'load_loss': load_loss.detach(),
			'expert_load': load.detach(),
			'expert_importance': importance.detach(),
		}
		return balance_loss, aux_dict

	def compute_state_prior_balance_loss(self, prior_out: Dict[str, torch.Tensor]):
		"""
		Keep the multi-scale selector from collapsing to a single scale.
		A small KL-to-uniform on alpha is enough in most cases.
		"""
		alpha = prior_out.get('state_alpha', prior_out.get('alpha'))
		if alpha is None:
			raise KeyError('state_alpha (or alpha) is missing from prior_out.')
		num_scales = alpha.numel()
		target = alpha.new_full((num_scales,), 1.0 / float(num_scales))
		alpha_loss = torch.sum(alpha * torch.log((alpha + 1e-8) / target))
		alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum()
		aux_dict = {
			'state_prior_balance_loss': alpha_loss.detach(),
			'state_alpha_entropy': alpha_entropy.detach(),
			'state_alpha': alpha.detach(),
		}
		return alpha_loss, aux_dict

	def construct_index(self, num: int):
		self.keys = torch.zeros(num, self.seq_len, self.c_in, device=self.device)
		self.values = torch.zeros(num, self.pred_len, self.dec_in, device=self.device)
		self.index = 0

	@torch.no_grad()
	def add_key_value(self, x_enc: torch.Tensor, y: torch.Tensor, index: torch.Tensor):
		self.keys[index, :, :] = x_enc
		self.values[index, :, :] = y
		self.index += x_enc.size(0)
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	@staticmethod
	def cosine_similarity(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
		if queries.dim() == 3:
			bsz, n_keys = queries.size(0), keys.size(0)
			queries = queries.reshape(bsz, -1)
			keys = keys.reshape(n_keys, -1)
		elif queries.dim() != 2:
			raise ValueError(f'Unsupported query shape: {queries.shape}')

		q_norm = F.normalize(queries, p=2, dim=-1)
		k_norm = F.normalize(keys, p=2, dim=-1)
		return torch.matmul(q_norm, k_norm.t())

	def retrieval(self, x: torch.Tensor, index: Optional[torch.Tensor]):
		if self.index == 0:
			raise RuntimeError('Retrieval index has not been constructed yet.')
		bsz = x.shape[0]
		k = min(self.retrieval_num, self.index)
		keys = self.keys[:self.index]
		values = self.values[:self.index]
		dis = self.cosine_similarity(x, keys)

		if self.training and index is not None:
			self_range = torch.arange(-self.seq_len, self.seq_len + 1, device=x.device).unsqueeze(0)
			invalid_index = (index.unsqueeze(1) + self_range) // self.retrieval_stride
			invalid_index = invalid_index.clamp(0, self.index - 1)
			row_idx = torch.arange(bsz, device=x.device).unsqueeze(1).expand_as(invalid_index)
			dis[row_idx, invalid_index] = -100.0

		dis_topk, indices_topk = torch.topk(dis, dim=1, k=k)
		probs_topk = torch.softmax(dis_topk, dim=1).unsqueeze(-1).unsqueeze(-1)
		retrieved_values = values[indices_topk]
		output = torch.sum(probs_topk * retrieved_values, dim=1)
		return output, dis_topk, 0

	def _forward_backbone(self, x: torch.Tensor, dec_input: Optional[torch.Tensor] = None):
		x_emb_hist = self.enc_embedding(x)
		dec_emb = self.dec_embedding(dec_input)
		enc_out, (h_n, c_n) = self.encoder(x_emb_hist)
		dec_out, _ = self.decoder(dec_emb, (h_n, c_n))
		dec_out = dec_out[:, -self.pred_len:, :]

		ctx, _ = self.xattn(dec_out, enc_out, enc_out)
		final_shared = self.post_norm(self.fuse_proj(torch.cat([dec_out, ctx], dim=-1)))

		router_logits, prior_out = self.router(x, final_shared)
		router_prob = torch.softmax(router_logits, dim=-1)
		topk_probs, topk_experts = torch.topk(router_prob, k=self.top_k_experts, dim=-1)
		head_mix_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

		moe_out = self.moe_head(final_shared, head_mix_weights=head_mix_weights, topk_experts=topk_experts)
		moe_out.update({
			'router_logits': router_logits,
			'router_prob': router_prob,
			'topk_experts': topk_experts,
			'topk_probs': head_mix_weights,
			'state_probs': prior_out['q_final'],
			'state_q_scales': prior_out['q_scales'],
			'state_scale_scores': prior_out['scale_scores'],
			'state_alpha': prior_out['alpha'],
		})
		return moe_out

	def _heuristic_alpha(self, sims: torch.Tensor):
		sim_mean = sims.mean(dim=-1)
		dynamic_alpha = (sim_mean - self.retrieval_tau) / (1.0 - self.retrieval_tau + 1e-8)
		dynamic_alpha = self.retrieval_alpha_max * dynamic_alpha.clamp(0.0, 1.0)
		return dynamic_alpha.view(-1, 1, 1)

	def _fuse_with_retrieval(self, x: torch.Tensor, point_pred: torch.Tensor, sample_ids: Optional[torch.Tensor], use_gate: bool):
		retrieval_results, sims, _ = self.retrieval(x, sample_ids)
		retrieval_pred = retrieval_results[:, :, :self.out_dim]
		if use_gate:
			beta = self.beta_gate(x_enc=x, base_pred=point_pred.detach(), ret_pred=retrieval_pred.detach(), sims=sims)
		else:
			beta = self._heuristic_alpha(sims)
		fused_point = (1.0 - beta) * point_pred + beta * retrieval_pred
		return fused_point, retrieval_pred, sims, beta

	@staticmethod
	def _store_retrieval_aux(aux_dict: Dict[str, torch.Tensor], sims: torch.Tensor, beta: torch.Tensor):
		aux_dict['beta_mean'] = beta.mean().detach()
		aux_dict['beta_max'] = beta.max().detach()
		aux_dict['sim_mean'] = sims.mean().detach()

	def forward(
		self,
		x: torch.Tensor,
		x_mark: Optional[torch.Tensor] = None,
		dec_input: Optional[torch.Tensor] = None,
		sample_ids: Optional[torch.Tensor] = None,
		route_labels: Optional[torch.Tensor] = None,
		mode: str = 'train',
		return_aux: bool = False,
	):
		_ = x_mark, route_labels
		if mode == 'gate_train':
			with torch.no_grad():
				out = self._forward_backbone(x=x, dec_input=dec_input)
		else:
			out = self._forward_backbone(x=x, dec_input=dec_input)

		point_pred = out['point_pred']
		total_aux_loss = point_pred.new_tensor(0.0)
		aux_dict: Dict[str, torch.Tensor] = {}

		if mode in {'gate_train', 'gate_valid'}:
			point_pred, retrieval_pred, sims, beta = self._fuse_with_retrieval(x, point_pred, sample_ids, use_gate=True)
			total_aux_loss = total_aux_loss + self.retrieval_beta_reg * beta.mean()
			self._store_retrieval_aux(aux_dict, sims, beta)
			out['retrieval_pred'] = retrieval_pred
			out['beta'] = beta
		elif mode == 'test' and hasattr(self, 'index') and self.index > 0:
			use_gate = bool(self.retrieval_gate_ready.item())
			point_pred, retrieval_pred, sims, beta = self._fuse_with_retrieval(x, point_pred, sample_ids, use_gate=use_gate)
			self._store_retrieval_aux(aux_dict, sims, beta)
			out['retrieval_pred'] = retrieval_pred
			out['beta'] = beta

		if mode in {'train', 'valid'}:
			balance_loss, balance_aux_dict = self.compute_sample_level_balance_loss(out['router_logits'])
			total_aux_loss = total_aux_loss + 0.1 * balance_loss
			aux_dict.update(balance_aux_dict)

			state_prior_loss, state_prior_aux_dict = self.compute_state_prior_balance_loss(out)
			total_aux_loss = total_aux_loss + self.state_prior_balance_reg * state_prior_loss
			aux_dict.update(state_prior_aux_dict)

		out['point_pred'] = point_pred
		out['total_aux_loss'] = total_aux_loss
		aux_dict['total_aux_loss'] = total_aux_loss.detach()
		aux_dict['router_prob'] = out['router_prob'].detach()
		aux_dict['topk_experts'] = out['topk_experts'].detach()
		aux_dict['state_probs'] = out['state_probs'].detach()
		aux_dict['state_alpha'] = out['state_alpha'].detach()
		self.latest_aux_dict = aux_dict

		if return_aux:
			return out, total_aux_loss, aux_dict
		return out, total_aux_loss
