#!/bin/bash

pred_lens=(5)
d_models=(64)
datasets=("Abilene" "Geant")
data_files=("Abilene_12_12_3000.csv" "Geant_23_23_3000.csv")
num_vars=(144 529)

configs=(
  "PMDformerConfig"
  "ITransformerConfig"
  "FEDformerConfig"
  "FeTSConfig"
  "HMformerConfig"
  "PatchTSTConfig"
  "TimesNetConfig"
  "WPMixerConfig"
  "P_sLSTMConfig"
  "xLSTMTimeConfig"
  "xlstm_mixerConfig"
)

for i in "${!datasets[@]}"
do
  dataset="${datasets[$i]}"
  data_file="${data_files[$i]}"
  var_dim="${num_vars[$i]}"

  for config in "${configs[@]}"
  do
    for pred in "${pred_lens[@]}"
    do
      for dm in "${d_models[@]}"
      do
        echo ">> Running baseline multi: config=${config}, dataset=${dataset}, data_file=${data_file}, vars=${var_dim}, pred_len=${pred}, d_model=${dm}"

        python "run_train_net_baseline_multi.py" \
          --config "$config" \
          --pred_len "$pred" \
          --d_model "$dm" \
          --dataset "$dataset" \
          --data_file "$data_file" \
          --epochs 200 \
          --patience 40 \
          --seq_len 96 \
          --rounds 1 \
          --logger "baseline_multi" \
          --loss_func "L1Loss" \
          --bs 32 \
          --seed 2026 \
          --retrain True \
          --enc_in "$var_dim" \
          --dec_in "$var_dim" \
          --c_out "$var_dim" \
          --out_dim "$var_dim" \
          --target_dim "$var_dim" \
          --target_cols "all" \
          --draw_root "./draw_multi"
      done
    done
  done
done
