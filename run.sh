#!/bin/bash

pred_lens=(12 96 192 336 720)

for len in $pred_lens
do
  echo "Running with pred_len=$len"
  python run_train.py --exp_name MLPConfig --retrain 1 --pred_len "$len"
done


#for len in $pred_lens
#do
#  echo "Running with pred_len=$len"
#  python run_train.py --exp_name TestConfig --retrain 1 --pred_len "$len"
#done