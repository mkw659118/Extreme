#!/bin/bash

python "run_train.py" \
  --config "PatchExtremeMemoryTransformerConfig" \
  --reservoir_sensor 'reservoir_stor_4001_sof24' \
  --d_model 256 \
  --use_memory False\
  --epochs 200\
  --patience 40\
  --train_volume 30000\
  --rounds 1\
  --oversampling 40\
  --use_memory True\
  --seq_weight 0\
  --data_model 'Almaden'
  
      
python "run_train.py" \
  --config "PatchExtremeMemoryTransformerConfig" \
  --reservoir_sensor 'reservoir_stor_4005_sof24' \
  --d_model 256 \
  --use_memory False\
  --epochs 200\
  --patience 40\
  --train_volume 40000\
  --rounds 1\
  --oversampling 40\
  --use_memory True\
  --seq_weight 0.4\
  --data_model 'Coyote'


python "run_train.py" \
  --config "PatchExtremeMemoryTransformerConfig" \
  --reservoir_sensor 'reservoir_stor_4007_sof24' \
  --d_model 256 \
  --use_memory False\
  --epochs 200\
  --patience 40\
  --train_volume 40000\
  --rounds 1\
  --oversampling 40\
  --use_memory True\
  --seq_weight 0\
  --data_model 'Lexington'
         
python "run_train.py" \
  --config "PatchExtremeMemoryTransformerConfig" \
  --reservoir_sensor 'reservoir_stor_4009_sof24' \
  --d_model 256 \
  --use_memory False\
  --epochs 200\
  --patience 40\
  --train_volume 40000\
  --rounds 1\
  --oversampling 40\
  --use_memory True\
  --seq_weight 0.4\
  --data_model 'Stevens_Creek'
      

python "run_train.py" \
  --config "PatchExtremeMemoryTransformerConfig" \
  --reservoir_sensor 'reservoir_stor_4011_sof24' \
  --d_model 256 \
  --use_memory False\
  --epochs 200\
  --patience 40\
  --train_volume 40000\
  --rounds 1\
  --oversampling 40\
  --use_memory True\
  --seq_weight 1.2\
  --data_model 'Vasona'
      