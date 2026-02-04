#!/bin/bash

reservoir_sensors=(
  "reservoir_stor_4001_sof24"
  "reservoir_stor_4005_sof24"
  "reservoir_stor_4007_sof24"
  "reservoir_stor_4009_sof24"
  "reservoir_stor_4011_sof24"
)

# pred_len 放在第一层
pred_lens=(8 72)


# use_memory 放在第二层
use_memories=(False True)

# 数据集放在第三层
# d_model 放在第四层
d_models=(8 16 32 64 128 256 512)

for pred in "${pred_lens[@]}"
do
  for um in "${use_memories[@]}"
  do
    for sensor in "${reservoir_sensors[@]}"
    do
      for dm in "${d_models[@]}"
      do
        echo ">> Running pred_len=${pred}, use_memory=${um}, sensor=${sensor}, d_model=${dm}"
        python "run_train.py" \
          --config "ExtremeLSTMConfig" \
          --reservoir_sensor "$sensor" \
          --pred_len "$pred" \
          --d_model "$dm" \
          --epochs 200 \
          --patience 40 \
          --train_volume 40000 \
          --rounds 1 \
          --revin False \
          --oversampling 40 \
          --use_memory "$um" \
          --loss_func 'L1Loss' \
          --bs 256 \
          --use_decoding False
      done
    done
  done
done

