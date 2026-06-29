#!/bin/bash

pred_lens=(5 10 15 20)
d_models=(256)
datasets=("Abilene" "Geant" "Seattle")
data_files=("Abilene_single.csv" "Geant_single.csv" "Seattle_single.csv")
configs=("NetStudentTPriorConfig" "NetGaussianPriorConfig")

for i in "${!datasets[@]}"
do
  dataset="${datasets[$i]}"
  data_file="${data_files[$i]}"

  for cfg in "${configs[@]}"
  do
    for pred in "${pred_lens[@]}"
    do
      for dm in "${d_models[@]}"
      do
        echo ">> Running prior distribution compare: config=${cfg}, dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

        python "run_train_DARNet.py" \
          --config "$cfg" \
          --pred_len "$pred" \
          --d_model "$dm" \
          --dataset "$dataset" \
          --data_file "$data_file" \
          --epochs 200 \
          --patience 40 \
          --seq_len 96 \
          --rounds 5 \
          --logger 'net_prior_distribution_compare' \
          --loss_func 'L1Loss' \
          --bs 32 \
          --seed 2026 \
          --retrain True \
          --use_state_prior True \
          --use_retrieval True \
          --use_missing_aware_encoding True \
          --num_experts 4 \
          --top_k_experts 2 \
          --retrieval_num 2 \
          --state_prior_scales "1,4,8,16" \
          --state_prior_include_seq_level True
      done
    done
  done
done
