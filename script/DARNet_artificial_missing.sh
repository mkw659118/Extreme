#!/bin/bash

pred_lens=(5 10 15 20)
d_models=(256)
missing_rates=(0.05 0.10 0.20)
configs=("NetConfig")

run_dataset() {
  local dataset="$1"
  local data_file="$2"

  for cfg in "${configs[@]}"
  do
    for rate in "${missing_rates[@]}"
    do
      rate_tag="${rate/./p}"
      for pred in "${pred_lens[@]}"
      do
        for dm in "${d_models[@]}"
        do
          echo ">> Running artificial missing: dataset=${dataset}, rate=${rate}, pred_len=${pred}, d_model=${dm}"

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
            --logger "DARNet_artificial_missing_${rate_tag}" \
            --loss_func 'L1Loss' \
            --bs 32 \
            --seed 2026 \
            --retrain True \
            --artificial_missing_rate "$rate" \
            --artificial_missing_seed 2026 \
            --artificial_missing_splits 'train,val,test' \
            --artificial_missing_target_only False
        done
      done
    done
  done
}

run_dataset 'Abilene' 'Abilene_single.csv'
run_dataset 'Geant' 'Geant_single.csv'
run_dataset 'Seattle' 'Seattle_single.csv'
