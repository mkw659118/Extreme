#!/bin/bash

## 预测长度列表
#pred_lens=(96 192 336 720)
#
## 模型配置列表
#exp_names=(MLP5Config SeasonalTrendModelConfig DFTDecomModelConfig TransformerLibraryConfig TransformerConfig)
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

## 预测长度列表
#pred_lens=(96 192 336 720)
#
#
#  for len in "${pred_lens[@]}"
#  do
#    echo "Running with exp_name=, pred_len=$len"
#    python run_mkw.py --exp_name "SeasonalTrendModelConfig" --retrain 1 --pred_len "$len" --revin True --logger mkw
#  done


#!/bin/bash

# 消融模式列表（外循环）
match_modes=("a" "ab" "ac" "bc" "abc")

# 预测长度列表（内循环）
pred_lens=(96 192 336 720)

for mode in "${match_modes[@]}"
do
  for len in "${pred_lens[@]}"
  do
    echo "Running with mode=$mode, pred_len=$len"
    python run_mkw.py \
      --exp_name "SeasonalTrendModelConfig" \
      --retrain 1 \
      --pred_len "$len" \
      --revin True \
      --mode "$mode" \
      --logger mkw
  done
done
