#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_ngr_baseline.py \
  --data_dir "${DATA_DIR}" \
  --save_dir ./outputs/checkpoints/ngr_like_gaussian_mos \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --topo_path "${TOPO_PATH}" \
  --use_validation

python scripts/infer_ngr_baseline.py \
  --data_dir "${DATA_DIR}" \
  --params_path ./outputs/checkpoints/ngr_like_gaussian_mos/traditional_ngr_error_covsigma_params.npz \
  --output_root ./outputs/predictions/ngr_like_gaussian_mos \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --topo_path "${TOPO_PATH}" \
  --split test

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/ngr_like_gaussian_mos \
  --target_root_glob "${DATA_ROOT_GLOB}" --split test \
  --out_dir ./outputs/metrics/ngr_like_gaussian_mos
