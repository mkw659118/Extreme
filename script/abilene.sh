#!/bin/bash

# 1) pred_len 第一层
pred_lens=(5)
d_models=(256 512)
# configs=("DLinearConfig" "NLinearConfig" "PMDformerConfig" "ITransformerConfig" "InformerConfig" "FEDformerConfig" "FeTSConfig" "HMformerConfig")
configs=("FEDformerConfig")

# for pred in "${pred_lens[@]}"
# do
#   for dm in "${d_models[@]}"
#   do
#     echo ">> Running config='NetConfig', pred_len=${pred}, d_model=${dm}"
#     python "run_train_net.py" \
#       --config "NetConfig" \
#       --pred_len "$pred" \
#       --d_model "$dm" \
#       --dataset 'Abilene' \
#       --data_file 'Abilene_12_12_3000.csv' \
#       --epochs 200 \
#       --patience 40 \
#       --rounds 5 \
#       --seq_len 96 \
#       --loss_func 'L1Loss' \
#       --bs 32 \
#       --retrain True
#   done
# done

for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_net_baseline.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Abilene' \
        --data_file 'Abilene_12_12_3000.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True
    done
  done
done

pred_lens=(5)
d_models=(16 32 64 128 256 512)
configs=("DLinearConfig" "NLinearConfig" "PMDformerConfig" "ITransformerConfig" "InformerConfig" "FEDformerConfig" "FeTSConfig" "HMformerConfig")

for pred in "${pred_lens[@]}"
do
  for dm in "${d_models[@]}"
  do
    echo ">> Running config='NetConfig', pred_len=${pred}, d_model=${dm}"
    python "run_train_net.py" \
      --config "NetConfig" \
      --pred_len "$pred" \
      --d_model "$dm" \
      --dataset 'Geant' \
      --data_file 'Geant_23_23_3000.csv' \
      --epochs 200 \
      --patience 40 \
      --rounds 5 \
      --seq_len 96 \
      --loss_func 'L1Loss' \
      --bs 32 \
      --retrain True \
      --target_col 1
  done
done

for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_net_baseline.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Geant' \
        --data_file 'Geant_23_23_3000.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True \
        --target_col 1
    done
  done
done

for pred in "${pred_lens[@]}"
do
  for dm in "${d_models[@]}"
  do
    echo ">> Running config='NetConfig', pred_len=${pred}, d_model=${dm}"
    python "run_train_net.py" \
      --config "NetConfig" \
      --pred_len "$pred" \
      --d_model "$dm" \
      --dataset 'Seattle' \
      --data_file 'Seattle.csv' \
      --epochs 200 \
      --patience 40 \
      --rounds 5 \
      --seq_len 96 \
      --loss_func 'L1Loss' \
      --bs 32 \
      --lr 0.00005 \
      --retrain True \
      --target_col 1
  done
done

for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_net_baseline.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Seattle' \
        --data_file 'Seattle.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --loss_func 'L1Loss' \
        --bs 32 \
        --lr 0.00005 \
        --retrain True \
        --target_col 1
    done
  done
done