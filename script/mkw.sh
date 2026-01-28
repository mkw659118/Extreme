#!/bin/bash
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
      --config "ExtremeLSTMConfig" \
      --reservoir_sensor "$sensor" \
      --pred_len "$pred" \
      --d_model 512\
      --epochs 200\
      --patience 40\
      --train_volume 40000\
      --rounds 1\
      --revin False\
      --oversampling 40\
      --use_memory False\
      --loss_func 'L1Loss'\
      --bs 256\
      --use_decoding False
  done
done






# 外循环：预测长度
for pred in "${pred_lens[@]}"
do
  # 内循环：数据集
  for sensor in "${reservoir_sensors[@]}"
  do
    echo ">> Running with pred_len=${pred}, reservoir_sensor=${sensor}"
    python "run_train.py" \
      --config "ExtremeLSTMConfig" \
      --reservoir_sensor "$sensor" \
      --pred_len "$pred" \
      --d_model 512\
      --epochs 200\
      --patience 40\
      --train_volume 40000\
      --rounds 1\
      --revin False\
      --oversampling 40\
      --use_memory True\
      --loss_func 'L1Loss'\
      --bs 256\
      --use_decoding False
  done
done
