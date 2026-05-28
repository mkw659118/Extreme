#!/bin/bash


# 1) pred_len 第一层
pred_lens=(5 10)
d_models=(256 512)

# for pred in "${pred_lens[@]}"
# do
#   for dm in "${d_models[@]}"
#   do
#     echo ">> Running config='NetConfig', pred_len=${pred}, d_model=${dm}"
#     python "run_train_net.py" \
#       --config "NetConfig" \
#       --pred_len "$pred" \
#       --d_model "$dm" \
#       --epochs 200 \
#       --patience 40 \
#       --rounds 5 \
#       --loss_func 'L1Loss' \
#       --bs 64 \
#       --retrain True
#   done
# done

# configs=("DLinearConfig" "NLinearConfig" "PMDformerConfig" "ITransformerConfig" "InformerConfig" "FEDformerConfig" "FeTSConfig" "HMformerConfig")
configs=("HMformerConfig")

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
        --epochs 2 \
        --patience 40 \
        --rounds 5 \
        --seq_len 96 \
        --loss_func 'L1Loss' \
        --bs 32 \
        --retrain True
    done
  done
done
