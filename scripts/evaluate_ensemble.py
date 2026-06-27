import argparse
import csv
import glob
import os
from datetime import datetime

import numpy as np


VARIABLES = ["q2m", "u10", "v10", "sp", "t2m"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate saved SERD ensemble samples against residual-error targets."
    )
    parser.add_argument("--sample_root", type=str, required=True)
    parser.add_argument("--target_root_glob", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./outputs/metrics/serd_v1")
    parser.add_argument("--train_count", type=int, default=1292)
    parser.add_argument("--valid_count", type=int, default=92)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"])
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
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


def fair_crps_ensemble(ens, obs):
    # ens: [K,H,W], obs: [H,W]
    k = ens.shape[0]
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)
    if k <= 1:
        return term1
    diffs = np.abs(ens[:, None, ...] - ens[None, :, ...])
    pair = np.sum(diffs, axis=(0, 1)) / (k * (k - 1))
    return term1 - 0.5 * pair


def load_ensemble(path):
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected [K,C,H,W] or [C,H,W], got {arr.shape}: {path}")
    return arr.astype(np.float32)


def target_file_for_day(day_dir, lead_hour):
    day_name = os.path.basename(day_dir.rstrip("/\\"))
    try:
        date_token = datetime.strptime(day_name, "%Y%m%d").strftime("%Y_%m_%d")
    except ValueError:
        date_token = datetime.strptime(day_name, "%Y-%m-%d-%H").strftime("%Y_%m_%d")
    candidates = [
        os.path.join(day_dir, f"{date_token}_{lead_hour}_err.npy"),
        os.path.join(day_dir, f"{date_token}_{lead_hour:02d}_err.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_days = select_split(
        sorted(glob.glob(args.target_root_glob)),
        args.split,
        args.train_count,
        args.valid_count,
    )
    if not target_days:
        raise RuntimeError(f"No target days selected for split={args.split}")

    n_vars = len(VARIABLES)
    se_sum = np.zeros(n_vars, dtype=np.float64)
    spread_sum = np.zeros(n_vars, dtype=np.float64)
    crps_sum = np.zeros(n_vars, dtype=np.float64)
    count = np.zeros(n_vars, dtype=np.float64)
    coverage_inside = {0.80: np.zeros(n_vars), 0.90: np.zeros(n_vars), 0.95: np.zeros(n_vars)}
    rank_hist = None
    missing = 0

    for target_day in target_days:
        day_name = os.path.basename(target_day.rstrip("/\\"))
        try:
            init_time = datetime.strptime(day_name, "%Y%m%d").replace(hour=9).strftime("%Y-%m-%d-%H")
        except ValueError:
            init_time = day_name
        sample_day = os.path.join(args.sample_root, init_time)
        for lead_hour in range(3, 73, 3):
            sample_path = os.path.join(sample_day, f"{init_time}_{lead_hour:02d}.npy")
            target_path = target_file_for_day(target_day, lead_hour)
            if not os.path.exists(sample_path) or not os.path.exists(target_path):
                missing += 1
                continue

            ens = load_ensemble(sample_path)[:, :, :args.height, :args.width]
            obs = np.load(target_path).astype(np.float32)[:, :args.height, :args.width]
            k = ens.shape[0]
            if rank_hist is None:
                rank_hist = np.zeros((n_vars, k + 1), dtype=np.float64)

            mean = np.mean(ens, axis=0)
            spread = np.std(ens, axis=0, ddof=1) if k > 1 else np.zeros_like(mean)
            for v in range(n_vars):
                err = mean[v] - obs[v]
                se_sum[v] += np.sum(err * err)
                spread_sum[v] += np.sum(spread[v])
                crps_sum[v] += np.sum(fair_crps_ensemble(ens[:, v], obs[v]))
                n = obs[v].size
                count[v] += n

                for nominal in coverage_inside:
                    alpha = 1.0 - nominal
                    lo = np.quantile(ens[:, v], alpha / 2.0, axis=0)
                    hi = np.quantile(ens[:, v], 1.0 - alpha / 2.0, axis=0)
                    coverage_inside[nominal][v] += np.sum((obs[v] >= lo) & (obs[v] <= hi))

                ranks = np.sum(ens[:, v] < obs[v][None, ...], axis=0)
                rank_hist[v] += np.bincount(ranks.ravel(), minlength=k + 1)

    rmse = np.sqrt(se_sum / np.maximum(count, 1.0))
    spread = spread_sum / np.maximum(count, 1.0)
    spread_rmse = spread / np.maximum(rmse, 1e-12)
    crps = crps_sum / np.maximum(count, 1.0)

    metrics_path = os.path.join(args.out_dir, f"{args.split}_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "rmse", "spread", "spread_rmse", "fcrps", "count"])
        for i, name in enumerate(VARIABLES):
            writer.writerow([name, rmse[i], spread[i], spread_rmse[i], crps[i], int(count[i])])

    coverage_path = os.path.join(args.out_dir, f"{args.split}_coverage.csv")
    with open(coverage_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nominal", "variable", "coverage", "coverage_error"])
        for nominal, inside in coverage_inside.items():
            for i, name in enumerate(VARIABLES):
                cov = inside[i] / max(count[i], 1.0)
                writer.writerow([nominal, name, cov, cov - nominal])

    rank_path = os.path.join(args.out_dir, f"{args.split}_rank_histogram.csv")
    if rank_hist is not None:
        with open(rank_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variable", "rank", "count"])
            for i, name in enumerate(VARIABLES):
                for rank, value in enumerate(rank_hist[i]):
                    writer.writerow([name, rank, value])

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved coverage to {coverage_path}")
    print(f"Saved rank histogram to {rank_path}")
    print(f"Missing sample/target lead cases: {missing}")


if __name__ == "__main__":
    main()

