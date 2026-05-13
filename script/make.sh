#!/bin/bash

reservoir_sensors=(
  "reservoir_stor_4001_sof24"
  "reservoir_stor_4005_sof24"
  "reservoir_stor_4007_sof24"
  "reservoir_stor_4009_sof24"
  "reservoir_stor_4011_sof24"
)


# 1) pred_len 第一层
pred_lens=(8 72)
d_models=(8 16 32 64 96 128 256 512)
# d_models=(128)

for pred in "${pred_lens[@]}"
do
  for sensor in "${reservoir_sensors[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running pred_len=${pred}, sensor=${sensor}, d_model=${dm}"
      python "run_train_last_qtopk.py" \
        --config "ExtremeLSTMMemoConfig" \
        --reservoir_sensor "$sensor" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --epochs 200 \
        --patience 40 \
        --train_volume 40000 \
        --rounds 1 \
        --oversampling 40 \
        --loss_func 'L1Loss' \
        --bs 128 \
        --retrain True
    done
  done
done

reservoir_sensors=(
  "reservoir_stor_4001_sof24"
  "reservoir_stor_4005_sof24"
  "reservoir_stor_4007_sof24"
  "reservoir_stor_4009_sof24"
  "reservoir_stor_4011_sof24"
)


# 1) pred_len 第一层
pred_lens=(72)
d_models=(32 64 512)
# d_models=(128)

for pred in "${pred_lens[@]}"
do
  for sensor in "${reservoir_sensors[@]}"
  do
    for dm in "${d_models[@]}"
    do
      echo ">> Running pred_len=${pred}, sensor=${sensor}, d_model=${dm}"
      python "run_train_last_no_student_t.py" \
        --config "ExtremeLSTMMemoConfig" \
        --reservoir_sensor "$sensor" \
        --pred_len "$pred" \
        --d_model "$dm" \
        --epochs 200 \
        --patience 40 \
        --train_volume 40000 \
        --rounds 1 \
        --oversampling 40 \
        --loss_func 'L1Loss' \
        --bs 256 \
        --retrain True
    done
  done
done

