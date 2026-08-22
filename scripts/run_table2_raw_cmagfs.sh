#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-./example_data/CMA_gfs_time_order_3_72/*[0-9]}"

python scripts/evaluate_raw_cmagfs.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --split test \
  --out_dir ./outputs/metrics/raw_cmagfs
