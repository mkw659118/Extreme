#!/bin/bash

pred_lens=(96 192 336 720)

for len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$len"
  python run_mkw.py --exp_name MLP1Config --retrain 1 --pred_len "$len" --revin True --logger mkw
done


for len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$len"
  python run_mkw.py --exp_name MLP2Config --retrain 1 --pred_len "$len" --revin True --logger mkw
done

for len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$len"
  python run_mkw.py --exp_name MLP3Config --retrain 1 --pred_len "$len" --revin True --logger mkw
done

