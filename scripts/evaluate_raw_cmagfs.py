import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np
from serd.paper.spec import LEAD_HOURS, SURFACE_CHANNEL_INDICES, file_candidates, find_existing, select_date_split


VARIABLES = ["q2m", "u10", "v10", "sp", "t2m"]


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate raw CMA-GFS surface forecasts against CMA-RRA analysis.")
    parser.add_argument("--data_root_glob", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs/metrics/raw_cmagfs")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_levels", type=int, default=9)
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    day_dirs = select_date_split(sorted(glob.glob(args.data_root_glob)), args.split)
    if not day_dirs:
        raise RuntimeError(f"No files selected for split={args.split}")

    shape = (len(LEAD_HOURS), len(VARIABLES))
    error_sum = np.zeros(shape, dtype=np.float64)
    abs_error_sum = np.zeros(shape, dtype=np.float64)
    se_sum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    missing = 0

    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir.rstrip("/\\"))
        date_token = datetime.strptime(day_name, "%Y%m%d").strftime("%Y_%m_%d")
        for lead_index, lead_hour in enumerate(LEAD_HOURS):
            fc_path = find_existing(file_candidates(day_dir, date_token, lead_hour))
            ana_path = find_existing(file_candidates(day_dir, date_token, lead_hour, "_analysis"))
            if fc_path is None or ana_path is None:
                missing += 1
                continue

            forecast = np.load(fc_path).astype(np.float32)
            analysis = np.load(ana_path).astype(np.float32)[:, :args.height, :args.width]
            surface_fc = forecast[list(SURFACE_CHANNEL_INDICES), :args.height, :args.width]
            err = surface_fc - analysis
            error_sum[lead_index] += np.sum(err, axis=(1, 2))
            abs_error_sum[lead_index] += np.sum(np.abs(err), axis=(1, 2))
            se_sum[lead_index] += np.sum(err * err, axis=(1, 2))
            count[lead_index] += analysis.shape[1] * analysis.shape[2]

    out_path = os.path.join(args.out_dir, f"{args.split}_lead_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lead_hour", "variable", "bias", "rmse", "crps", "count"])
        for lead_index, lead in enumerate(LEAD_HOURS):
            for variable_index, name in enumerate(VARIABLES):
                n = max(count[lead_index, variable_index], 1.0)
                writer.writerow([lead, name, error_sum[lead_index, variable_index] / n,
                                 np.sqrt(se_sum[lead_index, variable_index] / n),
                                 abs_error_sum[lead_index, variable_index] / n,
                                 int(count[lead_index, variable_index])])

    table_path = os.path.join(args.out_dir, f"{args.split}_table3_metrics.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "bias", "rmse", "crps", "count"])
        for variable_index, name in enumerate(VARIABLES):
            n = max(count[:, variable_index].sum(), 1.0)
            writer.writerow([name, error_sum[:, variable_index].sum() / n,
                             np.sqrt(se_sum[:, variable_index].sum() / n),
                             abs_error_sum[:, variable_index].sum() / n,
                             int(count[:, variable_index].sum())])

    print(f"Saved raw CMA-GFS lead metrics to {out_path}")
    print(f"Missing lead cases: {missing}")


if __name__ == "__main__":
    main()
