#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_table2_raw_cmagfs.sh
bash scripts/run_table2_gridleadbias.sh
bash scripts/run_table2_ngr_like_gaussian_mos.sh
bash scripts/run_table2_corrdiff.sh
bash scripts/run_table2_direct_diffusion_fcrps.sh
bash scripts/run_table2_twostage_no_fcrps.sh
bash scripts/run_table2_serd.sh
