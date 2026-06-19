#!/usr/bin/env bash
set -euo pipefail

pred_lens=(5)
d_models=(256)
datasets=("Abilene" "Geant")
data_files=("Abilene_single.csv" "Geant_single.csv")

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    data_file="${data_files[$i]}"

    for pred in "${pred_lens[@]}"; do
        for dm in "${d_models[@]}"; do
            echo ">> Running DATP-Net horizon-level MoE router: dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

            python "run_train_DARNet.py" \
                --config "DATPNetHorizonConfig" \
                --pred_len "${pred}" \
                --d_model "${dm}" \
                --dataset "${dataset}" \
                --data_file "${data_file}" \
                --epochs 200 \
                --patience 40 \
                --seq_len 96 \
                --rounds 1 \
                --logger "DATP_Net_horizon_router" \
                --loss_func "L1Loss" \
                --bs 32 \
                --seed 2026 \
                --retrain True \
                --num_experts 4 \
                --top_k_experts 2 \
                --retrieval_num 2 \
                --state_prior_scales "1,4,8,16" \
                --state_prior_include_seq_level True \
                --use_state_prior True \
                --use_retrieval True \
                --use_missing_aware_encoding True \
                --router_granularity "horizon" \
                --horizon_router_use_state_context True \
                --router_balance_weight 1.0 \
                --topk_coverage_weight 0.6 \
                --router_entropy_weight 0.004 \
                --topk_min_usage 0.12 \
                --moe_weight_balance_weight 2.0 \
                --moe_min_weight 0.08 \
                --moe_max_weight 0.55 \
                --topk_weight_floor_weight 1.0 \
                --topk_min_head_weight 0.18 \
                --horizon_diversity_weight 3.0 \
                --horizon_min_weight_std 0.04 \
                --horizon_max_cosine 0.75 \
                --router_temperature 0.9 \
                --router_train_noise_std 0.04 \
                --ensure_all_experts_in_topk False \
                --gate_epochs 0 \
                --pretrain_epochs 0
        done
    done
done
