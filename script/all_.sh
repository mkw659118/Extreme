#!/bin/bash


# 1) pred_len 第一层
pred_lens=(10 15 20)
d_models=(256)


# configs=("PMDformerConfig" "ITransformerConfig" "FEDformerConfig" "FeTSConfig" "HMformerConfig" "PatchTSTConfig" "WPMixerConfig" "P_sLSTMConfig" "TimesNetConfig")
configs=("TimesNetConfig")



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
        --data_file 'Abilene_single.csv' \
        --epochs 200 \
        --patience 40 \
        --seq_len 96 \
        --logger 'baseline' \
        --rounds 5 \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True
    done
  done
done


