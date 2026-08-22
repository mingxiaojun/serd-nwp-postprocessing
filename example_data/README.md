# Example CMA-GFS / CMA-RRA data

`CMA_gfs_time_order_3_72` contains two test-period initialization dates copied
from the paper dataset:

- `20241229`: 20 forecast/error/analysis triplets, leads 3--60 h
- `20241230`: 12 forecast/error/analysis triplets, leads 3--36 h

Each forecast array has shape `[45, 200, 200]` and stores one near-surface
variable followed by its eight corresponding upper-air variables, repeated for
the five target-variable groups. Each physical-unit error array and analysis
array has shape `[5, 200, 200]`.

The stored arrays are 200 × 200 before the spatial cropping/preprocessing step
used to obtain the 192 × 192 model domain.

Filename examples:

- forecast: `2024_12_29_48.npy`
- physical-unit error: `2024_12_29_48_err.npy`
- analysis: `2024_12_29__48_analysis.npy`

The `.npy` files are managed by Git LFS. Run `git lfs pull` if a clone contains
LFS pointer text instead of NumPy arrays. Validate the downloaded files with:

```bash
python scripts/validate_example_data.py
```

This subset is for file-format and data-loader checks only. It does not contain
training or validation dates and cannot reproduce the paper experiments.
