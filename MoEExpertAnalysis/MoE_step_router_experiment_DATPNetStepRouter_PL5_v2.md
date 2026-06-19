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

- Abilene: `checkpoints/datp_net_step/DatasetAbilene_Modeldatp_net_step_PL5_DM256_BS32_27cd3d6a_round_0.pt`
- GÉANT: `checkpoints/datp_net_step/DatasetGeant_Modeldatp_net_step_PL5_DM256_BS32_3cf79424_round_0.pt`

## Outputs

- token CSV: `MoEExpertAnalysis/data/moe_step_router_tokens_DATPNetStepRouter_PL5_v3.csv`
- usage CSV: `MoEExpertAnalysis/data/moe_step_router_usage_DATPNetStepRouter_PL5_v3.csv`
- step CSV: `MoEExpertAnalysis/data/moe_step_router_by_history_step_DATPNetStepRouter_PL5_v3.csv`
- usage figure: `MoEExpertAnalysis/figures/moe_step_router_usage_DATPNetStepRouter_PL5_v3.pdf`
- Top-K heatmap: `MoEExpertAnalysis/figures/moe_step_router_topk_heatmap_DATPNetStepRouter_PL5_v3.pdf`
- weight heatmap: `MoEExpertAnalysis/figures/moe_step_router_weight_heatmap_DATPNetStepRouter_PL5_v3.pdf`

## Expert Usage Summary

| dataset | pred_len | expert | topk_usage | mean_router_weight | mean_moe_weight | top1_usage |
| --- | --- | --- | --- | --- | --- | --- |
| Abilene | 5 | 0 | 0.282369 | 0.249983 | 0.249984 | 0.250000 |
| Abilene | 5 | 1 | 0.810700 | 0.250002 | 0.250019 | 0.250000 |
| Abilene | 5 | 2 | 0.445225 | 0.250008 | 0.249999 | 0.250000 |
| Abilene | 5 | 3 | 0.461707 | 0.250007 | 0.249998 | 0.250000 |
| GÉANT | 5 | 0 | 0.464503 | 0.238554 | 0.238557 | 0.242362 |
| GÉANT | 5 | 1 | 0.430876 | 0.240442 | 0.240286 | 0.241086 |
| GÉANT | 5 | 2 | 0.529327 | 0.253182 | 0.253297 | 0.254614 |
| GÉANT | 5 | 3 | 0.575294 | 0.267821 | 0.267860 | 0.261937 |

## Reading Guide

- `Top-K Activation` is the fraction of history-step tokens where an expert appears in the Top-K set.
- `Mean Router Weight` is the dense router probability averaged over all history-step tokens.
- `Mean MoE Weight` is the sparse normalized Top-K weight actually applied to expert outputs in the MoE fusion.
- If step-level routing is working, the heatmaps should vary across history steps instead of showing one constant sample-level choice.
- The auxiliary balance loss encourages all experts to receive traffic, while the entropy term keeps router weights from becoming exactly identical.