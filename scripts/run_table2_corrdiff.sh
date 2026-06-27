#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/path/to/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_corrdiff_stage1.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --save_dir ./outputs/checkpoints/corrdiff

python scripts/train_corrdiff_diffusion.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --train_count 1292 \
  --valid_count 92 \
  --save_dir ./outputs/checkpoints/corrdiff
