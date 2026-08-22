# SERD NWP Post-processing

This repository contains the recommended code layout for the paper experiment:
Systematic-error-corrected Residual Diffusion (SERD) for multivariate probabilistic post-processing of deterministic near-surface NWP forecasts.

## Recommended Experiment

Use a single experiment id: `serd_v1`.

Unified data split:

- Training: 2020-01-01 to 2023-09-30
- Validation: 2023-10-01 to 2023-12-31
- Testing: 2024-01-01 to 2024-12-31

The scripts select these calendar intervals explicitly. Sample counts are checked after file matching/QC and never define the split.

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
example_data/
  CMA_gfs_time_order_3_72/  two-day Git LFS example dataset
```

## Example Dataset

This repository includes a two-day example dataset under
`example_data/CMA_gfs_time_order_3_72/`. The files are stored with Git LFS, so
clone the repository with Git LFS installed or run `git lfs pull` after cloning.

The example contains 32 complete forecast/error/analysis triplets:

- `20241229`: 20 lead times from 3 to 60 h
- `20241230`: 12 lead times from 3 to 36 h

The stored arrays are 200 × 200 before the spatial cropping/preprocessing step
used to obtain the 192 × 192 model domain.

Run the example-data integrity check from the repository root:

```bash
python scripts/validate_example_data.py
```

All code and configuration defaults resolve the source data from:

```text
./example_data/CMA_gfs_time_order_3_72/*[0-9]
```

The example is intended for inspecting the file contract and testing data
loading. It contains test-period dates only and is not sufficient for model
training, strict paper-sample validation, or reproducing the reported results.
Set `DATA_ROOT_GLOB` or pass `--data_root_glob` to use the complete dataset.

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

All entries use the same explicit calendar split shown above.

## Recommended Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the recommended SERD pipeline with the complete paper dataset:

```bash
export DATA_ROOT_GLOB="/path/to/full/CMA_gfs_time_order_3_72/*[0-9]"
export DATA_DIR="./data"
export TOPO_PATH="./data/topo_data_Normalization.npy"
bash scripts/run_serd_v1_pipeline.sh
```

The pipeline performs:

1. Train the stage-1 deterministic mean/systematic-error model on train split and select best checkpoint on validation split.
2. Infer stage-1 corrections for all splits.
3. Build stage-2 residual-error data.
4. Train the stage-2 VE-SDE residual diffusion model with score loss + fCRPS and select best checkpoint on validation split.
5. Generate 16-member final forecast ensembles in physical units.

Because the original exclusion list is unavailable, regenerate and verify it
against the complete dataset before training:

```bash
python scripts/audit_paper_dataset.py --strict
```

This writes included/excluded CSV manifests and checks the manuscript counts `32560/2208/8685`.

Paper evaluation is strict by default. The evaluator reconstructs the canonical source manifest from
forecast, error, and analysis files; requires the manuscript sample count for the selected split; and
fails if any method output is missing. Ensemble filenames use two-digit lead hours (`_03.npy`,
`_06.npy`, ..., `_72.npy`). The complete Table 2 runner additionally checks all probabilistic methods
against the same 8,685 test-sample keys before Figure 8 is evaluated.

Evaluate test ensembles:

```bash
python scripts/evaluate_ensemble.py \
  --sample_root ./outputs/predictions/serd_v1/stage2_serd \
  --target_root_glob "./example_data/CMA_gfs_time_order_3_72/*[0-9]" \
  --split test \
  --out_dir ./outputs/metrics/serd_v1
```

## Data availability

The complete CMA-GFS reforecast and CMA-RRA reanalysis datasets used in this
study are not redistributed. A small two-day example is included through Git
LFS to document the forecast-analysis pairing, array shapes, channel order, and
filename convention. Users need to obtain the complete source data from the CMA
Earth System Modeling and Prediction Centre (CEMC), China Meteorological
Administration, to reproduce training and the paper results.


## Checkpoint Naming

Recommended names:

- Stage 1 best: `reg_sysbias_best_serd_v1_stage1.pth`
- Stage 1 final: `reg_sysbias_final_serd_v1_stage1.pth`
- Stage 2 latest: `vesde_physcond_serd_v1_stage2_fcrps_latest_full.pth`
- Stage 2 best: `vesde_physcond_serd_v1_stage2_fcrps_best.pth`
- Stage 2 final: `vesde_physcond_serd_v1_stage2_fcrps_final.pth`

## Baseline

The NGR baseline uses the same explicit calendar split and the same final physical-field evaluator.

## Paper-locked metric contract

- Table 3 and Figures 4--8 use ordinary empirical ensemble CRPS, with the pairwise term divided by `2 K^2`; fair CRPS is used only as the neural training regularizer.
- Ensemble spread uses sample standard deviation (`ddof=1`).
- Coverage error is `abs(actual - nominal)`.
- Rank histograms have 17 bins for the 16-member ensembles.
- Figure 8 stores the difference as `CorrDiff - SERD`, matching the manuscript caption.
