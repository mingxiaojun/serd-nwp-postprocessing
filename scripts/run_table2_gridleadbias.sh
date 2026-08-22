#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-./example_data/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"

python scripts/train_gridleadbias.py \
  --data_dir "${DATA_DIR}" \
  --save_dir ./outputs/checkpoints/gridleadbias \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --use_validation \
  --batch_size 2 \
  --eval_batch_size 2

python scripts/infer_gridleadbias.py \
  --params_path ./outputs/checkpoints/gridleadbias/allvars_gridlead_bias_params.npz \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --output_root ./outputs/predictions/gridleadbias --split test

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/gridleadbias \
  --target_root_glob "${DATA_ROOT_GLOB}" --split test \
  --out_dir ./outputs/metrics/gridleadbias
