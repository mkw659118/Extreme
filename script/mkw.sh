#!/bin/bash

# 定义需要跑的 reservoir 数据集
reservoir_sensors=(
  "reservoir_stor_4001_sof24"
  "reservoir_stor_4005_sof24"
  "reservoir_stor_4007_sof24"
  "reservoir_stor_4009_sof24"
  "reservoir_stor_4011_sof24"
)

# 定义预测长度
pred_lens=(8 72)

# 外循环：预测长度
for pred in "${pred_lens[@]}"
do
  # 内循环：数据集
  for sensor in "${reservoir_sensors[@]}"
  do
    echo ">> Running with pred_len=${pred}, reservoir_sensor=${sensor}"
    python "run_train.py" \
      --config "TransformerConfig" \
      --reservoir_sensor "$sensor" \
      --pred_len "$pred" \
      --revin True
  done
done
