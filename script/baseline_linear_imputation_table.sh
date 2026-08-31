#!/usr/bin/env bash
set -euo pipefail

# Two-stage experiment:
#   incomplete history -> window-local linear interpolation -> baseline forecast
#
# Environment overrides, for example:
#   ROUNDS=1 EPOCHS=2 MODELS="PMDformerConfig PatchTSTConfig" bash script/baseline_linear_imputation_table.sh

PRED_LENS_TEXT="${PRED_LENS:-5}"
D_MODELS_TEXT="${D_MODELS:-256}"
ROUNDS="${ROUNDS:-5}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-40}"
SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_TEXT="${MODELS:-PMDformerConfig HMformerConfig FeTSConfig TimesNetConfig ITransformerConfig FEDformerConfig PatchTSTConfig WPMixerConfig P_sLSTMConfig xLSTMTimeConfig xlstm_mixerConfig}"

read -r -a pred_lens <<< "${PRED_LENS_TEXT}"
read -r -a d_models <<< "${D_MODELS_TEXT}"
read -r -a configs <<< "${MODELS_TEXT}"

run_dataset() {
  local dataset="$1"
  local data_file="$2"

  for cfg in "${configs[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
      for d_model in "${d_models[@]}"; do
        echo ">> Random missing 5% + linear interpolation: ${dataset}/${cfg}/PL${pred_len}"
        "${PYTHON_BIN}" run_train_net_baseline_linear.py \
          --config "${cfg}" \
          --dataset "${dataset}" \
          --data_file "${data_file}" \
          --pred_len "${pred_len}" \
          --d_model "${d_model}" \
          --seq_len "${SEQ_LEN}" \
          --rounds "${ROUNDS}" \
          --epochs "${EPOCHS}" \
          --patience "${PATIENCE}" \
          --bs "${BATCH_SIZE}" \
          --device "${DEVICE}" \
          --loss_func L1Loss \
          --seed "${SEED}" \
          --retrain True \
          --record True \
          --skip_startup_cleanup True \
          --logger two_stage_linear_random_5 \
          --experiment_tag two_stage_linear_random_5 \
          --artificial_missing_pattern random_point \
          --artificial_missing_rate 0.05 \
          --artificial_missing_seed "${SEED}" \
          --artificial_missing_splits train,val,test \
          --artificial_missing_target_only False

        echo ">> Structured time-block missing 20% + linear interpolation: ${dataset}/${cfg}/PL${pred_len}"
        "${PYTHON_BIN}" run_train_net_baseline_linear.py \
          --config "${cfg}" \
          --dataset "${dataset}" \
          --data_file "${data_file}" \
          --pred_len "${pred_len}" \
          --d_model "${d_model}" \
          --seq_len "${SEQ_LEN}" \
          --rounds "${ROUNDS}" \
          --epochs "${EPOCHS}" \
          --patience "${PATIENCE}" \
          --bs "${BATCH_SIZE}" \
          --device "${DEVICE}" \
          --loss_func L1Loss \
          --seed "${SEED}" \
          --retrain True \
          --record True \
          --skip_startup_cleanup True \
          --logger two_stage_linear_structured_20 \
          --experiment_tag two_stage_linear_structured_20 \
          --artificial_missing_pattern time_block \
          --artificial_missing_rate 0.20 \
          --artificial_missing_seed "${SEED}" \
          --artificial_missing_splits train,val,test \
          --artificial_missing_target_only False \
          --artificial_missing_block_length 12 \
          --artificial_missing_column_rate 1.0
      done
    done
  done
}

run_dataset Abilene Abilene_single.csv
run_dataset Geant Geant_single.csv

for pred_len in "${pred_lens[@]}"; do
  for d_model in "${d_models[@]}"; do
    "${PYTHON_BIN}" utils/build_two_stage_linear_table.py \
      --pred-len "${pred_len}" \
      --d-model "${d_model}" \
      --strict
  done
done
