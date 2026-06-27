# SERD NWP Post-processing

This repository contains the recommended code layout for the paper experiment:
Systematic-error-corrected Residual Diffusion (SERD) for multivariate probabilistic post-processing of deterministic near-surface NWP forecasts.

## Recommended Experiment

Use a single experiment id: `serd_v1`.

Unified data split:

- Train: first `1292` initialization days.
- Validation: next `92` initialization days.
- Test: all remaining initialization days.

The target variables are `[q2m, u10, v10, sp, t2m]` and the lead times are 3-72 h at 3 h intervals.

## Layout

```text
serd/
  data/      datasets and normalizers
  models/    stage-1 mean U-Net and stage-2 conditional diffusion model
  utils/     split helpers
scripts/
  run_table2_*.sh
  train_stage1_mean.py
  infer_stage1_mean.py
  build_stage2_residuals.py
  train_stage2_serd.py
  infer_stage2_serd.py
  evaluate_ensemble.py
  train_ngr_baseline.py
  infer_ngr_baseline.py
configs/
  serd_v1.yaml
  table2/
docs/
  EXPERIMENT_DESIGN.md
```

## Table 2 Experiments

Each Table 2 method has an independent config under `configs/table2/` and a matching entry script under `scripts/`.

| Paper method | Config | Entry |
| --- | --- | --- |
| CMA-GFS | `configs/table2/raw_cmagfs.yaml` | `scripts/run_table2_raw_cmagfs.sh` |
| GridLeadBias | `configs/table2/gridleadbias.yaml` | `scripts/run_table2_gridleadbias.sh` |
| NGR-like Gaussian MOS | `configs/table2/ngr_like_gaussian_mos.yaml` | `scripts/run_table2_ngr_like_gaussian_mos.sh` |
| CorrDiff | `configs/table2/corrdiff.yaml` | `scripts/run_table2_corrdiff.sh` |
| Direct diffusion + fCRPS | `configs/table2/direct_diffusion_fcrps.yaml` | `scripts/run_table2_direct_diffusion_fcrps.sh` |
| Two-stage w/o fCRPS | `configs/table2/twostage_no_fcrps.yaml` | `scripts/run_table2_twostage_no_fcrps.sh` |
| SERD | `configs/table2/serd.yaml` | `scripts/run_table2_serd.sh` |

All entries use the same chronological split: `1292` train days, `92` validation days, and all remaining days for test.

## Recommended Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the recommended SERD pipeline:

```bash
export DATA_ROOT_GLOB="/path/to/CMA_gfs_time_order_3_72/*[0-9]"
export DATA_DIR="./data"
export TOPO_PATH="./data/topo_data_Normalization.npy"
bash scripts/run_serd_v1_pipeline.sh
```

The pipeline performs:

1. Train the stage-1 deterministic mean/systematic-error model on train split and select best checkpoint on validation split.
2. Infer stage-1 corrections for all splits.
3. Build stage-2 residual-error data.
4. Train the stage-2 VE-SDE residual diffusion model with score loss + fCRPS and select best checkpoint on validation split.
5. Generate test-split residual-error ensembles.

Evaluate test ensembles:

```bash
python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/serd_v1/stage2_serd \
  --target_root_glob "./data/stage2_residuals_serd_v1/*[0-9]" \
  --split test \
  --out_dir ./outputs/metrics/serd_v1
```

## Checkpoint Naming

Recommended names:

- Stage 1 best: `reg_sysbias_best_serd_v1_stage1.pth`
- Stage 1 final: `reg_sysbias_final_serd_v1_stage1.pth`
- Stage 2 latest: `vesde_physcond_serd_v1_stage2_fcrps_latest_full.pth`
- Stage 2 best: `vesde_physcond_serd_v1_stage2_fcrps_best.pth`
- Stage 2 final: `vesde_physcond_serd_v1_stage2_fcrps_final.pth`

## Baseline

The NGR baseline scripts in `scripts/train_ngr_baseline.py` and `scripts/infer_ngr_baseline.py` use the same `1292/92/test` split.
