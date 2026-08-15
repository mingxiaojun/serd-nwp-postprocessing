"""Evaluate final physical-unit forecast ensembles for Table 3 and Figures 4--8."""
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

import numpy as np

from serd.paper.metrics import absolute_coverage_error, empirical_crps, rank_histogram
from serd.paper.spec import LEAD_HOURS, SURFACE_VARIABLES, file_candidates, find_existing, select_date_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_root", required=True, help="Final physical ensembles [16,5,H,W]")
    parser.add_argument("--target_root_glob", default="/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]")
    parser.add_argument("--out_dir", default="./outputs/metrics")
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    return parser


def load_ensemble(path: Path) -> np.ndarray:
    value = np.load(path).astype(np.float32)
    if value.ndim != 4 or value.shape[1] != 5:
        raise ValueError(f"Expected final physical ensemble [K,5,H,W], got {value.shape}: {path}")
    if value.shape[0] != 16:
        raise ValueError(f"Paper evaluation requires K=16, got K={value.shape[0]}: {path}")
    return value


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(LEAD_HOURS), len(SURFACE_VARIABLES))
    error_sum = np.zeros(shape, dtype=np.float64)
    se_sum = np.zeros(shape, dtype=np.float64)
    spread_sum = np.zeros(shape, dtype=np.float64)
    crps_sum = np.zeros(shape, dtype=np.float64)
    inside80 = np.zeros(shape, dtype=np.int64)
    inside90 = np.zeros(shape, dtype=np.int64)
    counts = np.zeros(shape, dtype=np.int64)
    rank_counts = np.zeros((len(SURFACE_VARIABLES), 17), dtype=np.int64)
    missing = 0

    for day_text in select_date_split(sorted(glob.glob(args.target_root_glob)), args.split):
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        for lead_index, lead in enumerate(LEAD_HOURS):
            sample_path = Path(args.sample_root) / init_token / f"{init_token}_{lead:02d}.npy"
            analysis_path = find_existing(file_candidates(day_dir, date_token, lead, "_analysis"))
            if not sample_path.is_file() or analysis_path is None:
                missing += 1
                continue
            ens = load_ensemble(sample_path)[:, :, :args.height, :args.width]
            obs = np.load(analysis_path).astype(np.float32)[:, :args.height, :args.width]
            mean = ens.mean(axis=0)
            spread = ens.std(axis=0, ddof=1)
            crps = empirical_crps(ens, obs, member_axis=0)
            for index, variable in enumerate(SURFACE_VARIABLES):
                del variable
                error = mean[index] - obs[index]
                error_sum[lead_index, index] += error.sum()
                se_sum[lead_index, index] += np.square(error).sum()
                spread_sum[lead_index, index] += spread[index].sum()
                crps_sum[lead_index, index] += crps[index].sum()
                lo80, hi80 = np.quantile(ens[:, index], (.10, .90), axis=0)
                lo90, hi90 = np.quantile(ens[:, index], (.05, .95), axis=0)
                inside80[lead_index, index] += np.count_nonzero((obs[index] >= lo80) & (obs[index] <= hi80))
                inside90[lead_index, index] += np.count_nonzero((obs[index] >= lo90) & (obs[index] <= hi90))
                counts[lead_index, index] += obs[index].size
                rank_counts[index] += rank_histogram(ens[:, index], obs[index])

    if not np.any(counts):
        raise RuntimeError("No matched physical ensembles and analysis files were found")
    rows: list[dict[str, object]] = []
    for lead_index, lead in enumerate(LEAD_HOURS):
        for variable_index, variable in enumerate(SURFACE_VARIABLES):
            count = int(counts[lead_index, variable_index])
            if count == 0:
                continue
            actual80 = inside80[lead_index, variable_index] / count
            actual90 = inside90[lead_index, variable_index] / count
            rows.append({
                "lead_hour": lead, "variable": variable,
                "bias": float(error_sum[lead_index, variable_index] / count),
                "rmse": float(np.sqrt(se_sum[lead_index, variable_index] / count)),
                "spread": float(spread_sum[lead_index, variable_index] / count),
                "crps": float(crps_sum[lead_index, variable_index] / count),
                "coverage_error_80": absolute_coverage_error(actual80, .80),
                "coverage_error_90": absolute_coverage_error(actual90, .90),
                "count": count,
            })
    with (out_dir / f"{args.split}_lead_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{args.split}_table3_metrics.csv").open("w", newline="") as handle:
        fieldnames = ("variable", "bias", "rmse", "spread", "crps", "coverage_error_80", "coverage_error_90", "count")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variable_index, variable in enumerate(SURFACE_VARIABLES):
            count = int(counts[:, variable_index].sum())
            actual80 = inside80[:, variable_index].sum() / count
            actual90 = inside90[:, variable_index].sum() / count
            writer.writerow({
                "variable": variable,
                "bias": float(error_sum[:, variable_index].sum() / count),
                "rmse": float(np.sqrt(se_sum[:, variable_index].sum() / count)),
                "spread": float(spread_sum[:, variable_index].sum() / count),
                "crps": float(crps_sum[:, variable_index].sum() / count),
                "coverage_error_80": absolute_coverage_error(actual80, .80),
                "coverage_error_90": absolute_coverage_error(actual90, .90),
                "count": count,
            })
    with (out_dir / f"{args.split}_rank_histogram.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variable", "rank", "count"))
        for variable_index, variable in enumerate(SURFACE_VARIABLES):
            for rank, count in enumerate(rank_counts[variable_index]):
                writer.writerow((variable, rank, int(count)))
    print(f"lead_variable_rows={len(rows)} missing_samples={missing}")


if __name__ == "__main__":
    main()
