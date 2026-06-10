#!/bin/bash

pred_lens=(5 10 15 20)
d_models=(256)
configs=("NetConfig")
expert_nums=(1 2 4 6 8)

datasets=("Abilene" "Geant" "Seattle")
data_files=("Abilene_single.csv" "Geant_single.csv" "Seattle_single.csv")

for cfg in "${configs[@]}"
do
  for data_idx in "${!datasets[@]}"
  do
    dataset="${datasets[$data_idx]}"
    data_file="${data_files[$data_idx]}"

    for pred in "${pred_lens[@]}"
    do
      for dm in "${d_models[@]}"
      do
        for experts in "${expert_nums[@]}"
        do
          if [ "$experts" -lt 2 ]; then
            topk=1
          else
            topk=2
          fi

          echo ">> Num experts=${experts}, top_k=${topk}, config=${cfg}, dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

          python "run_train_DARNet.py" \
            --config "$cfg" \
            --pred_len "$pred" \
            --d_model "$dm" \
            --dataset "$dataset" \
            --data_file "$data_file" \
            --epochs 200 \
            --patience 40 \
            --seq_len 96 \
            --rounds 5 \
            --logger 'DARNet_hp_num_experts' \
            --loss_func 'L1Loss' \
            --bs 32 \
            --seed 2026 \
            --retrain True \
            --use_retrieval True \
            --use_state_prior True \
            --use_missing_aware_encoding True \
            --state_num 4 \
            --num_experts "$experts" \
            --top_k_experts "$topk" \
            --retrieval_num 2 \
            --state_prior_scales "1,4,8,16" \
            --state_prior_include_seq_level True
        done
      done
    done
  done
done
