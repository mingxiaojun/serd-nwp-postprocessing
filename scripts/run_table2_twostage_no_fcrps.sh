#!/usr/bin/env bash
set -euo pipefail

STAGE2_ROOT_GLOB="${STAGE2_ROOT_GLOB:-./data/stage2_residuals_serd_v1/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_twostage_no_fcrps.py \
  --data_root_glob "${STAGE2_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --lambda_fcrps 0.0 \
  --resume "" \
  --save_dir ./outputs/checkpoints/twostage_no_fcrps

python scripts/infer_twostage_no_fcrps.py \
  --data_root_glob "${STAGE2_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --split test \
  --ckpt_path ./outputs/checkpoints/twostage_no_fcrps/vesde_physcond_twostage_no_fcrps_best.pth \
  --output_root ./outputs/predictions/twostage_no_fcrps

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/twostage_no_fcrps \
  --target_root_glob "${STAGE2_ROOT_GLOB}" \
  --split test \
  --train_count 1292 \
  --valid_count 92 \
  --out_dir ./outputs/metrics/twostage_no_fcrps
