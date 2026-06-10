#!/bin/bash

pred_lens=(5 10 15 20)
d_models=(256)
configs=("NetConfig")

scale_sets=(
  "1"
  "1,4"
  "1,4,8"
  "1,4,8,16"
  "1,4,8,16,32"
  "1,4,8,16"
)

include_seq_levels=(
  "True"
  "True"
  "True"
  "True"
  "True"
  "False"
)

datasets=("Abilene" "Geant" "Seattle")
data_files=("Abilene_single.csv" "Geant_single.csv" "Seattle_single.csv")

for cfg in "${configs[@]}"
do
  for data_idx in "${!datasets[@]}"
  do
    dataset="${datasets[$data_idx]}"
    data_file="${data_files[$data_idx]}"

    for pred in "${pred_lens[@]}"
    do
      for dm in "${d_models[@]}"
      do
        for scale_idx in "${!scale_sets[@]}"
        do
          scales="${scale_sets[$scale_idx]}"
          include_seq="${include_seq_levels[$scale_idx]}"

          echo ">> State prior scales=${scales}, include_seq=${include_seq}, config=${cfg}, dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

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
            --logger 'DARNet_hp_state_prior_scales' \
            --loss_func 'L1Loss' \
            --bs 32 \
            --seed 2026 \
            --retrain True \
            --use_retrieval True \
            --use_state_prior True \
            --use_missing_aware_encoding True \
            --state_num 4 \
            --num_experts 4 \
            --top_k_experts 2 \
            --retrieval_num 2 \
            --state_prior_scales "$scales" \
            --state_prior_include_seq_level "$include_seq"
        done
      done
    done
  done
done
