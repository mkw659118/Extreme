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
d_models=(256)
# d_models=(128)
configs=(
  "P_sLSTMConfig"
  "xLSTMTimeConfig"
  "xlstm_mixerConfig"
  "WPMixerConfig"
  "PatchTSTConfig"
  "FEDformerConfig"
)

for cfg in "${configs[@]}"
do
  for pred in "${pred_lens[@]}"
  do
    for sensor in "${reservoir_sensors[@]}"
    do
      for dm in "${d_models[@]}"
      do
        echo ">> Running config=${cfg}, pred_len=${pred}, sensor=${sensor}, d_model=${dm}"
        python "run_train.py" \
          --config "$cfg" \
          --reservoir_sensor "$sensor" \
          --pred_len "$pred" \
          --d_model "$dm" \
          --epochs 200 \
          --patience 40 \
          --dataset 'water' \
          --train_volume 40000 \
          --rounds 1 \
          --logger 'water_baseline' \
          --oversampling 40 \
          --loss_func 'L1Loss' \
          --bs 256 \
          --retrain True
      done
    done
  done
done
