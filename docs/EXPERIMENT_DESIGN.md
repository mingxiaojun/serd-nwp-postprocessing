# Experiment Design

The recommended experiment id is `serd_v1`.

SERD follows the paper design:

1. Estimate the predictable systematic component of the CMA-GFS forecast error with a supervised deterministic model.
2. Construct a corrected field and a residual-error target.
3. Train a conditional VE-SDE diffusion model on the residual-error distribution.
4. Generate finite residual-error ensembles and evaluate reliability with CRPS, spread/RMSE, rank histograms, and prediction-interval coverage.

The unified split is:

- Train: 2020-01-01 through 2023-09-30.
- Validation: 2023-10-01 through 2023-12-31.
- Test: 2024-01-01 through 2024-12-31.

All training, inference, baselines, and evaluation scripts select this split by date. The reported 32,560/2,208/8,685 sample counts are post-QC checks, not slicing indices.

The 45 forecast channels contain five nine-channel blocks: one near-surface field followed by its eight pressure-level fields at 925, 850, 700, 500, 300, 200, 150, and 100 hPa. The model therefore receives five surface fields plus a `5 x 8` upper-air tensor.

Stage 1 uses raw physical total error (`analysis - forecast`) and the five loss weights `0.10/0.50/0.20/0.15/0.03`. Stage 2 uses `total error - stage-1 error` and fits a separate scaler from training residuals only.

## Table 2 Mapping

The release folder separates the paper comparison rows into formal configs and executable entries:

| Paper method | Experiment id | Main entry |
| --- | --- | --- |
| CMA-GFS | `raw_cmagfs` | `scripts/run_table2_raw_cmagfs.sh` |
| GridLeadBias | `gridleadbias` | `scripts/run_table2_gridleadbias.sh` |
| NGR-like Gaussian MOS | `ngr_like_gaussian_mos` | `scripts/run_table2_ngr_like_gaussian_mos.sh` |
| CorrDiff | `corrdiff` | `scripts/run_table2_corrdiff.sh` |
| Direct diffusion + fCRPS | `direct_diffusion_fcrps` | `scripts/run_table2_direct_diffusion_fcrps.sh` |
| Two-stage w/o fCRPS | `twostage_no_fcrps` | `scripts/run_table2_twostage_no_fcrps.sh` |
| SERD | `serd_v1` | `scripts/run_table2_serd.sh` |

`serd_v1` remains the only recommended final method. The other Table 2 entries are kept as controlled ablations or baselines with their own checkpoint and output directories.
