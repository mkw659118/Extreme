#!/bin/bash

pred_lens=(5)
d_models=(256)
datasets=("Abilene" "Geant")
data_files=("Abilene_12_12_300.csv" "Geant_23_23_3000.csv")

for i in "${!datasets[@]}"
do
  dataset="${datasets[$i]}"
  data_file="${data_files[$i]}"

  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running DATP-Net multi: dataset=${dataset}, data_file=${data_file}, pred_len=${pred}, d_model=${dm}"

      python "run_train_DARNet_multi.py" \
        --config "DATPNetMultiConfig" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset "$dataset" \
        --data_file "$data_file" \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 1 \
        --logger "DATP_Net_multi" \
        --loss_func "L1Loss" \
        --bs 32 \
        --seed 2026 \
        --retrain True \
        --num_experts 4 \
        --top_k_experts 2 \
        --retrieval_num 2 \
        --state_prior_scales "1,4,8,16" \
        --state_prior_include_seq_level True \
        --state_prior_value_groups_per_channel 3 \
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
