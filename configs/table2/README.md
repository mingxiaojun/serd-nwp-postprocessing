# Table 2 Experiment Configs

These files map each comparison method in Table 2 of the paper to a concrete code entry point.

The unified split is:

- Train: first 1292 initialization days.
- Validation: next 92 initialization days.
- Test: all remaining initialization days.

Use `scripts/run_table2_*.sh` as the executable entry points.

`serd_v1` is the recommended final method. The other configs reproduce the Table 2 baselines and ablations with independent experiment ids, checkpoint directories, prediction directories, and metric directories.
