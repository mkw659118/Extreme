# Top-K MoE Expert Specialization Visualization

## Objective

This experiment checks whether the Top-K MoE router in DATP-Net forms expert specialization on Abilene and Geant.

## Setup

- Datasets: Abilene, Geant
- Prediction lengths: 5
- Number of experts: 4
- Top-K experts: 2
- Checkpoint round: 0
- Split used for analysis: test set
- Routing source: Student-T state prior responsibilities -> router -> Top-K experts

## Collected Routing Signals

- `router_prob`: dense probability over experts for each test sample.
- `topk_experts`: experts selected by Top-K routing.
- `state_probs`: Student-T state prior responsibility vector.
- `top1_expert`: expert with the largest router probability.
- `routing_entropy`: normalized entropy of expert probabilities; lower means sharper routing.

## Visualizations

- [moe_expert_usage_DATPNet_PL5](MoEExpertAnalysis/figures/moe_expert_usage_DATPNet_PL5.pdf)
- [moe_state_expert_alignment_DATPNet_PL5](MoEExpertAnalysis/figures/moe_state_expert_alignment_DATPNet_PL5.pdf)
- [moe_feature_by_topk_expert_DATPNet_PL5](MoEExpertAnalysis/figures/moe_feature_by_topk_expert_DATPNet_PL5.pdf)
- [moe_representative_samples_DATPNet_PL5](MoEExpertAnalysis/figures/moe_representative_samples_DATPNet_PL5.pdf)

## Summary Metrics

| dataset | pred_len | sample_count | mean_routing_entropy | expert_state_mutual_information | dominant_expert_ratio |
| --- | --- | --- | --- | --- | --- |
| Abilene | 5 | 596 | 0.999999 | 0.170213 | 0.681208 |
| Geant | 5 | 596 | 0.999999 | 0.00608518 | 0.989933 |

## Interpretation Guide

- Expert usage shows whether the router collapses to a small number of experts or uses multiple experts.
- State-expert alignment shows whether Student-T latent states map preferentially to different experts.
- Feature-by-expert boxplots show whether experts receive samples with different volatility, future change, or routing certainty.
- Representative samples show the input patterns most confidently assigned to each expert.

Stronger specialization is indicated by non-uniform state-expert heatmaps, distinct feature distributions by expert, and clear representative-pattern differences.