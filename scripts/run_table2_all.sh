#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_table2_raw_cmagfs.sh
bash scripts/run_table2_gridleadbias.sh
bash scripts/run_table2_ngr_like_gaussian_mos.sh
bash scripts/run_table2_corrdiff.sh
bash scripts/run_table2_direct_diffusion_fcrps.sh
bash scripts/run_table2_serd.sh
bash scripts/run_table2_twostage_no_fcrps.sh

python scripts/evaluate_spatial_crps.py \
  --target_root_glob "${DATA_ROOT_GLOB:-/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]}"
