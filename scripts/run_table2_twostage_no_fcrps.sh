#!/usr/bin/env bash
set -euo pipefail

STAGE2_ROOT_GLOB="${STAGE2_ROOT_GLOB:-./data/stage2_residuals_serd_v1/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_twostage_no_fcrps.py \
  --data_root_glob "${STAGE2_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --err_scaler_path "${DATA_DIR}/scalers_stage2_residual_zscore_train.pkl" \
  --lambda_fcrps 0.0 \
  --resume "" \
  --save_dir ./outputs/checkpoints/twostage_no_fcrps

python scripts/infer_twostage_no_fcrps.py \
  --data_root_glob "${STAGE2_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --err_scaler_path "${DATA_DIR}/scalers_stage2_residual_zscore_train.pkl" \
  --split test \
  --ckpt_path ./outputs/checkpoints/twostage_no_fcrps/vesde_physcond_twostage_no_fcrps_best.pth \
  --output_root ./outputs/predictions/twostage_no_fcrps

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/twostage_no_fcrps \
  --target_root_glob "${DATA_ROOT_GLOB:-./example_data/CMA_gfs_time_order_3_72/*[0-9]}" \
  --split test \
  --out_dir ./outputs/metrics/twostage_no_fcrps
