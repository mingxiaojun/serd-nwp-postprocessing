import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np


VARIABLES = ["q2m", "u10", "v10", "sp", "t2m"]


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate raw CMA-GFS surface forecasts against CMA-RRA analysis.")
    parser.add_argument("--data_root_glob", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs/metrics/raw_cmagfs")
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num_levels", type=int, default=9)
    return parser


def select_split(paths, split, train_count, valid_count):
    train_end = train_count
    valid_end = train_count + valid_count
    if split == "train":
        return paths[:train_end]
    if split == "valid":
        return paths[train_end:valid_end]
    if split == "test":
        return paths[valid_end:]
    return paths


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    day_dirs = select_split(
        sorted(glob.glob(args.data_root_glob)),
        args.split,
        args.train_count,
        args.valid_count,
    )
    if not day_dirs:
        raise RuntimeError(f"No files selected for split={args.split}")

    se_sum = np.zeros(len(VARIABLES), dtype=np.float64)
    count = np.zeros(len(VARIABLES), dtype=np.float64)
    missing = 0

    for day_dir in day_dirs:
        day_name = os.path.basename(day_dir.rstrip("/\\"))
        date_token = datetime.strptime(day_name, "%Y%m%d").strftime("%Y_%m_%d")
        for lead_hour in range(3, 73, 3):
            fc_candidates = [
                os.path.join(day_dir, f"{date_token}_{lead_hour}.npy"),
                os.path.join(day_dir, f"{date_token}_{lead_hour:02d}.npy"),
            ]
            ana_candidates = [
                os.path.join(day_dir, f"{date_token}__{lead_hour}_analysis.npy"),
                os.path.join(day_dir, f"{date_token}__{lead_hour:02d}_analysis.npy"),
            ]
            fc_path = next((p for p in fc_candidates if os.path.exists(p)), None)
            ana_path = next((p for p in ana_candidates if os.path.exists(p)), None)
            if fc_path is None or ana_path is None:
                missing += 1
                continue

            forecast = np.load(fc_path).astype(np.float32)
            analysis = np.load(ana_path).astype(np.float32)[:, :args.height, :args.width]
            surface_fc = forecast[:: args.num_levels, :args.height, :args.width]
            err = surface_fc - analysis
            se_sum += np.sum(err * err, axis=(1, 2))
            count += analysis.shape[1] * analysis.shape[2]

    rmse = np.sqrt(se_sum / np.maximum(count, 1.0))
    out_path = os.path.join(args.out_dir, f"{args.split}_raw_cmagfs_rmse.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "rmse", "count"])
        for i, name in enumerate(VARIABLES):
            writer.writerow([name, rmse[i], int(count[i])])

    print(f"Saved raw CMA-GFS RMSE to {out_path}")
    print(f"Missing lead cases: {missing}")


if __name__ == "__main__":
    main()
