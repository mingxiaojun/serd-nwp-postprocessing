# Table 2 Experiment Configs

These files map each comparison method in Table 2 of the paper to a concrete code entry point.

The unified split is:

- Train: 2020-01-01 through 2023-09-30.
- Validation: 2023-10-01 through 2023-12-31.
- Test: 2024-01-01 through 2024-12-31.

Use `scripts/run_table2_*.sh` as the executable entry points.

`serd_v1` is the recommended final method. The other configs reproduce the Table 2 baselines and ablations with independent experiment ids, checkpoint directories, prediction directories, and metric directories.
