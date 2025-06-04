#!/bin/bash

# pred_lens=(12 96 192 336 720)
pred_lens=(96)

for len in "${pred_lens[@]}"
do
  echo "Running with pred_len=$len"
  python run_mkw.py --exp_name MLP2Config --retrain 1 --pred_len "$len" --revin False
done
