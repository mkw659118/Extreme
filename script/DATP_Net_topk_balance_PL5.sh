#!/bin/bash

pred_lens=(5)
d_models=(256)
datasets=("Abilene" "Geant")
data_files=("Abilene_single.csv" "Geant_single.csv")

for i in "${!datasets[@]}"
do
  dataset="${datasets[$i]}"
  data_file="${data_files[$i]}"

  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running DATP-Net Top-K balance: dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

      python "run_train_DARNet.py" \
        --config "DATPNetConfig" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset "$dataset" \
        --data_file "$data_file" \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 1 \
        --logger "DATP_Net_topk_balance" \
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
        --router_balance_weight 0.1 \
        --topk_coverage_weight 0.35 \
        --topk_min_usage 0.12 \
        --router_temperature 1.0 \
        --router_train_noise_std 0.03 \
        --ensure_all_experts_in_topk True
    done
  done
done
