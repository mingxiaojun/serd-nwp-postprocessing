# Experiment Design

The recommended experiment id is `serd_v1`.

SERD follows the paper design:

1. Estimate the predictable systematic component of the CMA-GFS forecast error with a supervised deterministic model.
2. Construct a corrected field and a residual-error target.
3. Train a conditional VE-SDE diffusion model on the residual-error distribution.
4. Generate finite residual-error ensembles and evaluate reliability with CRPS, spread/RMSE, rank histograms, and prediction-interval coverage.

The unified split is:

- Train: first `1292` initialization days.
- Validation: next `92` initialization days.
- Test: all remaining initialization days.

All training, inference, NGR baseline, and evaluation scripts should use this same split.

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
