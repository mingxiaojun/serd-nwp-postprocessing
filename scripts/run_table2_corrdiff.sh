#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_corrdiff_stage1.py \
  --data_root_glob "${DATA_ROOT_GLOB}" --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" --save_dir ./outputs/checkpoints/corrdiff

python scripts/infer_stage1_mean.py \
  --data_root_glob "${DATA_ROOT_GLOB}" --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" --target analysis --split all \
  --ckpt_path ./outputs/checkpoints/corrdiff/reg_sysbias_best_corrdiff_stage1.pth \
  --output_root ./outputs/predictions/corrdiff/stage1_analysis

python scripts/build_corrdiff_residuals.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --stage1_prediction_root ./outputs/predictions/corrdiff/stage1_analysis \
  --analysis_scaler_path "${DATA_DIR}/scalers_ana_zscore_two_step_unet_train.pkl" \
  --residual_scaler_path "${DATA_DIR}/scalers_corrdiff_residual_zscore_train.pkl" \
  --output_root ./data/corrdiff_residuals

python scripts/train_corrdiff_diffusion.py \
  --data_root_glob './data/corrdiff_residuals/*[0-9]' --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --err_scaler_path "${DATA_DIR}/scalers_corrdiff_residual_zscore_train.pkl" \
  --save_dir ./outputs/checkpoints/corrdiff

python scripts/infer_stage2_serd.py \
  --data_root_glob './data/corrdiff_residuals/*[0-9]' --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" --split test \
  --err_scaler_path "${DATA_DIR}/scalers_corrdiff_residual_zscore_train.pkl" \
  --ckpt_path ./outputs/checkpoints/corrdiff/corrdiff_score_only_best_full.pth \
  --output_root ./outputs/predictions/corrdiff

python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/corrdiff \
  --target_root_glob "${DATA_ROOT_GLOB}" --split test \
  --out_dir ./outputs/metrics/corrdiff
