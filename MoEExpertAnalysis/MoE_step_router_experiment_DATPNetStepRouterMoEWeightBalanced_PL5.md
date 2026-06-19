# Step-Level MoE Routing Analysis

## Setup

- Model: DATP-Net step-level router (`datp_net_step`)
- Prediction length: 5
- Datasets: Abilene, GÉANT
- Experts: 4
- Top-K: 2
- Routing grain: each history step has its own router distribution and Top-K expert set.
- Split for analysis: test set.

## Checkpoints

- Abilene: `checkpoints/datp_net_step/DatasetAbilene_Modeldatp_net_step_PL5_DM256_BS32_c6e3edb0_round_0.pt`
- GÉANT: `checkpoints/datp_net_step/DatasetGeant_Modeldatp_net_step_PL5_DM256_BS32_2da7d320_round_0.pt`

## Outputs

- token CSV: `MoEExpertAnalysis/data/moe_step_router_tokens_DATPNetStepRouterMoEWeightBalanced_PL5.csv`
- usage CSV: `MoEExpertAnalysis/data/moe_step_router_usage_DATPNetStepRouterMoEWeightBalanced_PL5.csv`
- step CSV: `MoEExpertAnalysis/data/moe_step_router_by_history_step_DATPNetStepRouterMoEWeightBalanced_PL5.csv`
- usage figure: `MoEExpertAnalysis/figures/moe_step_router_usage_DATPNetStepRouterMoEWeightBalanced_PL5.pdf`
- Top-K heatmap: `MoEExpertAnalysis/figures/moe_step_router_topk_heatmap_DATPNetStepRouterMoEWeightBalanced_PL5.pdf`
- weight heatmap: `MoEExpertAnalysis/figures/moe_step_router_weight_heatmap_DATPNetStepRouterMoEWeightBalanced_PL5.pdf`

## Expert Usage Summary

| dataset | pred_len | expert | topk_usage | mean_router_weight | mean_moe_weight | top1_usage |
| --- | --- | --- | --- | --- | --- | --- |
| Abilene | 5 | 0 | 0.520833 | 0.311017 | 0.311409 | 0.302083 |
| Abilene | 5 | 1 | 0.520833 | 0.209433 | 0.209424 | 0.218750 |
| Abilene | 5 | 2 | 0.479167 | 0.272335 | 0.272522 | 0.260242 |
| Abilene | 5 | 3 | 0.479167 | 0.207215 | 0.206644 | 0.218925 |
| GÉANT | 5 | 0 | 0.373095 | 0.184318 | 0.184819 | 0.142757 |
| GÉANT | 5 | 1 | 0.357540 | 0.182932 | 0.180376 | 0.225077 |
| GÉANT | 5 | 2 | 0.642460 | 0.238402 | 0.239926 | 0.110616 |
| GÉANT | 5 | 3 | 0.626905 | 0.394347 | 0.394879 | 0.521550 |

## Reading Guide

- `Top-K Activation` is the fraction of history-step tokens where an expert appears in the Top-K set.
- `Mean Router Weight` is the dense router probability averaged over all history-step tokens.
- `Mean MoE Weight` is the sparse normalized Top-K weight actually applied to expert outputs in the MoE fusion.
- If step-level routing is working, the heatmaps should vary across history steps instead of showing one constant sample-level choice.
- The auxiliary balance loss encourages all experts to receive traffic, while the entropy term keeps router weights from becoming exactly identical.