#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/path/to/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"

python scripts/train_gridleadbias.py \
  --data_dir "${DATA_DIR}" \
  --save_dir ./outputs/checkpoints/gridleadbias \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --train_count 1292 \
  --valid_count 92 \
  --use_validation \
  --batch_size 2 \
  --eval_batch_size 2
