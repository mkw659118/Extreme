#!/bin/bash

## 预测长度列表
#pred_lens=(96 192 336 720)
#
## 模型配置列表
#exp_names=(MLPConfig MLP2Config MLP3Config MLP4Config)
#
## 双重循环
#for exp in "${exp_names[@]}"
#do
#  for len in "${pred_lens[@]}"
#  do
#    echo "Running with exp_name=$exp, pred_len=$len"
#    python run_mkw.py --exp_name "$exp" --retrain 1 --pred_len "$len" --revin True --logger mkw
#  done
#done

# 预测长度列表
pred_lens=(96 192 336 720)


  for len in "${pred_lens[@]}"
  do
    echo "Running with exp_name=, pred_len=$len"
    python run_mkw.py --exp_name "DFTDecomModelConfig" --retrain 1 --pred_len "$len" --revin True --logger mkw
  done