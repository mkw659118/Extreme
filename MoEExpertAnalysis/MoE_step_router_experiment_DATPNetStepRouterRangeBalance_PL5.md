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

- Abilene: `checkpoints/datp_net_step/DatasetAbilene_Modeldatp_net_step_PL5_DM256_BS32_751dcb0f_round_0.pt`
- GÉANT: `checkpoints/datp_net_step/DatasetGeant_Modeldatp_net_step_PL5_DM256_BS32_471ca906_round_0.pt`

## Outputs

- token CSV: `MoEExpertAnalysis/data/moe_step_router_tokens_DATPNetStepRouterRangeBalance_PL5.csv`
- usage CSV: `MoEExpertAnalysis/data/moe_step_router_usage_DATPNetStepRouterRangeBalance_PL5.csv`
- step CSV: `MoEExpertAnalysis/data/moe_step_router_by_history_step_DATPNetStepRouterRangeBalance_PL5.csv`
- usage figure: `MoEExpertAnalysis/figures/moe_step_router_usage_DATPNetStepRouterRangeBalance_PL5.pdf`
- Top-K heatmap: `MoEExpertAnalysis/figures/moe_step_router_topk_heatmap_DATPNetStepRouterRangeBalance_PL5.pdf`
- weight heatmap: `MoEExpertAnalysis/figures/moe_step_router_weight_heatmap_DATPNetStepRouterRangeBalance_PL5.pdf`

## Expert Usage Summary

| dataset | pred_len | expert | topk_usage | mean_router_weight | mean_moe_weight | top1_usage |
| --- | --- | --- | --- | --- | --- | --- |
| Abilene | 5 | 0 | 0.999965 | 0.000022 | 0.000022 | 0.000000 |
| Abilene | 5 | 1 | 0.500000 | 0.499978 | 0.499989 | 0.500000 |
| Abilene | 5 | 2 | 0.000000 | 0.000022 | 0.000000 | 0.000000 |
| Abilene | 5 | 3 | 0.500035 | 0.499978 | 0.499989 | 0.500000 |
| GÉANT | 5 | 0 | 0.000017 | 0.000076 | 0.000000 | 0.000000 |
| GÉANT | 5 | 1 | 0.500192 | 0.496298 | 0.496334 | 0.496330 |
| GÉANT | 5 | 2 | 0.994425 | 0.000082 | 0.000081 | 0.000000 |
| GÉANT | 5 | 3 | 0.505366 | 0.503545 | 0.503585 | 0.503670 |

## Reading Guide

- `Top-K Activation` is the fraction of history-step tokens where an expert appears in the Top-K set.
- `Mean Router Weight` is the dense router probability averaged over all history-step tokens.
- `Mean MoE Weight` is the sparse normalized Top-K weight actually applied to expert outputs in the MoE fusion.
- If step-level routing is working, the heatmaps should vary across history steps instead of showing one constant sample-level choice.
- The auxiliary balance loss encourages all experts to receive traffic, while the entropy term keeps router weights from becoming exactly identical.