#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/path/to/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_direct_diffusion_fcrps.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --lambda_fcrps 0.5 \
  --resume "" \
  --save_dir ./outputs/checkpoints/direct_diffusion_fcrps

python scripts/infer_direct_diffusion_fcrps.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --split test \
  --ckpt_path ./outputs/checkpoints/direct_diffusion_fcrps/vesde_physcond_direct_diffusion_fcrps_best.pth \
  --output_root ./outputs/predictions/direct_diffusion_fcrps

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/direct_diffusion_fcrps \
  --target_root_glob "${DATA_ROOT_GLOB}" \
  --split test \
  --train_count 1292 \
  --valid_count 92 \
  --out_dir ./outputs/metrics/direct_diffusion_fcrps
