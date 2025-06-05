#!/bin/bash

pred_lens=(96 192 336 720)


for len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$len"
  python run_mkw.py --exp_name MLPConfig --retrain 1 --pred_len "$len" --revin False --logger mkw
done
