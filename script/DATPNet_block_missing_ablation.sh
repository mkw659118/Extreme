#!/usr/bin/env bash
set -euo pipefail

# DATP-Net component ablation under identical structured time-block missingness.
# Run from any directory. Environment variables below can override the defaults,
# e.g. ROUNDS=1 EPOCHS=2 DATASETS="Geant" bash script/DATPNet_block_missing_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-DATPNetConfig}"
DATASETS_TEXT="${DATASETS:-Abilene Geant}"
PRED_LENS_TEXT="${PRED_LENS:-5 10 15 20}"
D_MODEL="${D_MODEL:-256}"
SEQ_LEN="${SEQ_LEN:-96}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-40}"
ROUNDS="${ROUNDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-2026}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
GATE_EPOCHS="${GATE_EPOCHS:-5}"
MISSING_RATE="${MISSING_RATE:-0.20}"
BLOCK_LENGTH="${BLOCK_LENGTH:-12}"
COLUMN_RATE="${COLUMN_RATE:-1.0}"
MISSING_SPLITS="${MISSING_SPLITS:-train,val,test}"

read -r -a DATASETS_ARRAY <<< "${DATASETS_TEXT}"
read -r -a PRED_LENS_ARRAY <<< "${PRED_LENS_TEXT}"

variants=(full wo_moe wo_retrieval wo_state_prior wo_missing_aware_encoding)

data_file_for() {
  case "$1" in
    Abilene) echo "Abilene_single.csv" ;;
    Geant) echo "Geant_single.csv" ;;
    Seattle) echo "Seattle_single.csv" ;;
    *)
      echo "Unsupported dataset '$1'. Add its CSV mapping in data_file_for()." >&2
      return 1
      ;;
  esac
}

run_variant() {
  local dataset="$1"
  local data_file="$2"
  local pred_len="$3"
  local variant="$4"

  local use_missing=True
  local use_state=True
  local use_retrieval=True
  local num_experts=4
  local top_k_experts=2

  case "${variant}" in
    full) ;;
    wo_moe)
      num_experts=1
      top_k_experts=1
      ;;
    wo_retrieval)
      use_retrieval=False
      ;;
    wo_state_prior)
      use_state=False
      ;;
    wo_missing_aware_encoding)
      use_missing=False
      ;;
    *)
      echo "Unknown variant '${variant}'" >&2
      return 1
      ;;
  esac

  local rate_tag="${MISSING_RATE//./p}"
  local tag="block${rate_tag}_len${BLOCK_LENGTH}_cols${COLUMN_RATE}_${variant}"

  echo ">> dataset=${dataset} pred_len=${pred_len} variant=${variant} tag=${tag}"

  "${PYTHON_BIN}" run_train_DATPNet.py \
    --config "${CONFIG}" \
    --dataset "${dataset}" \
    --data_file "${data_file}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${pred_len}" \
    --d_model "${D_MODEL}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --rounds "${ROUNDS}" \
    --bs "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --retrain True \
    --skip_startup_cleanup True \
    --logger "DATPNet_block_missing_ablation_${variant}" \
    --experiment_tag "${tag}" \
    --loss_func L1Loss \
    --artificial_missing_pattern time_block \
    --artificial_missing_rate "${MISSING_RATE}" \
    --artificial_missing_block_length "${BLOCK_LENGTH}" \
    --artificial_missing_column_rate "${COLUMN_RATE}" \
    --artificial_missing_seed "${SEED}" \
    --artificial_missing_splits "${MISSING_SPLITS}" \
    --artificial_missing_target_only False \
    --use_missing_aware_encoding "${use_missing}" \
    --use_state_prior "${use_state}" \
    --use_retrieval "${use_retrieval}" \
    --state_num 4 \
    --num_experts "${num_experts}" \
    --top_k_experts "${top_k_experts}" \
    --retrieval_num 2 \
    --state_prior_scales "1,4,8,16" \
    --state_prior_include_seq_level True \
    --pretrain_epochs "${PRETRAIN_EPOCHS}" \
    --gate_epochs "${GATE_EPOCHS}"
}

for dataset in "${DATASETS_ARRAY[@]}"; do
  data_file="$(data_file_for "${dataset}")"
  for pred_len in "${PRED_LENS_ARRAY[@]}"; do
    for variant in "${variants[@]}"; do
      run_variant "${dataset}" "${data_file}" "${pred_len}" "${variant}"
    done
  done
done

echo ">> All DATP-Net block-missing ablation runs completed."
