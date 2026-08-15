#!/usr/bin/env bash
set -euo pipefail

EXP_ID="serd_v1"
DATA_ROOT_GLOB="${DATA_ROOT_GLOB:-/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]}"
DATA_DIR="${DATA_DIR:-./data}"
TOPO_PATH="${TOPO_PATH:-./data/topo_data_Normalization.npy}"

python scripts/train_stage1_mean.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --save_dir "./outputs/checkpoints/${EXP_ID}"

python scripts/infer_stage1_mean.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --split all \
  --target error \
  --ckpt_path "./outputs/checkpoints/${EXP_ID}/reg_sysbias_best_serd_v1_stage1.pth" \
  --output_root "./outputs/predictions/${EXP_ID}/stage1_mean"

python scripts/build_stage2_residuals.py \
  --data_root_glob "${DATA_ROOT_GLOB}" \
  --total_error_scaler_path "${DATA_DIR}/scalers_err_zscore_two_step_unet_train.pkl" \
  --residual_scaler_path "${DATA_DIR}/scalers_stage2_residual_zscore_train.pkl" \
  --stage1_prediction_root "./outputs/predictions/${EXP_ID}/stage1_mean" \
  --output_root "./data/stage2_residuals_${EXP_ID}"

python scripts/train_stage2_serd.py \
  --data_root_glob "./data/stage2_residuals_${EXP_ID}/*[0-9]" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --err_scaler_path "${DATA_DIR}/scalers_stage2_residual_zscore_train.pkl" \
  --resume "" \
  --save_dir "./outputs/checkpoints/${EXP_ID}"

python scripts/infer_stage2_serd.py \
  --data_root_glob "./data/stage2_residuals_${EXP_ID}/*[0-9]" \
  --data_dir "${DATA_DIR}" \
  --topo_path "${TOPO_PATH}" \
  --split test \
  --err_scaler_path "${DATA_DIR}/scalers_stage2_residual_zscore_train.pkl" \
  --ckpt_path "./outputs/checkpoints/${EXP_ID}/vesde_physcond_serd_v1_stage2_fcrps_best.pth" \
  --output_root "./outputs/predictions/${EXP_ID}/stage2_serd"

python scripts/evaluate_ensemble.py \
  --sample_root "./outputs/predictions/${EXP_ID}/stage2_serd" \
  --target_root_glob "${DATA_ROOT_GLOB}" \
  --split test \
  --out_dir "./outputs/metrics/${EXP_ID}"
