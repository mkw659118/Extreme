#!/bin/bash

pred_lens=(5 10 15 20)
d_models=(256)
configs=("NetConfig")
retrieval_topks=(1 2 3 5 8)

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
        for topk in "${retrieval_topks[@]}"
        do
          echo ">> Retrieval topK=${topk}, config=${cfg}, dataset=${dataset}, pred_len=${pred}, d_model=${dm}"

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
            --logger 'DARNet_hp_retrieval_topk' \
            --loss_func 'L1Loss' \
            --bs 32 \
            --seed 2026 \
            --retrain True \
            --use_retrieval True \
            --use_state_prior True \
            --use_missing_aware_encoding True \
            --state_num 4 \
            --num_experts 4 \
            --top_k_experts 2 \
            --retrieval_num "$topk" \
            --state_prior_scales "1,4,8,16" \
            --state_prior_include_seq_level True
        done
      done
    done
  done
done
