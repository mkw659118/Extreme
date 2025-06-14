# !/bin/bash

pred_lens=(12 96 192 336 720)
exp_names=(MLPConfig RNNConfig LSTMConfig GRUConfig CrossformerConfig TimesNetConfig)

# 跑对手的模型
for exp in "${exp_names[@]}"
do
  for len in "${pred_lens[@]}"
  do
    echo "run_train.py --exp_name $exp --retrain 1 --pred_len $len"
    python -u run_train.py --exp_name "$exp" --retrain 1 --pred_len "$len"
  done
done

# 跑自己的模型
exp_names=(TimeSeriesConfig)
for len in "${pred_lens[@]}"
  do
    echo "run_train.py --exp_name $exp --retrain 1 --pred_len $len"
    python -u run_train.py --exp_name "$exp" --retrain 1 --pred_len "$len"
  done