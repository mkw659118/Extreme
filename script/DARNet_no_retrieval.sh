#!/bin/bash

# 1) pred_len 第一层
pred_lens=(5 10 15 20)
d_models=(256)

configs=("NetConfig")


for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_DARNet.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Abilene' \
        --data_file 'Abilene_single.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --logger 'DARNet_no_retrieval' \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True \
        --use_retrieval False 
    done
  done
done



for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_DARNet.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Geant' \
        --data_file 'Geant_single.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --logger 'DARNet_no_retrieval' \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True \
        --use_retrieval False 
    done
  done
done



for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running config=${cfg}, pred_len=${pred}, d_model=${dm}"

      python "run_train_DARNet.py" \
        --config "$cfg" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --dataset 'Seattle' \
        --data_file 'Seattle_single.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --rounds 5 \
        --logger 'DARNet_no_retrieval' \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True \
        --use_retrieval False 
    done
  done
done