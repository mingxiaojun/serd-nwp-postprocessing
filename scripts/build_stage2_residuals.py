"""Build physical SERD residual targets and a train-only residual scaler."""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from serd.paper.spec import (LEAD_HOURS, SURFACE_CHANNEL_INDICES, file_candidates,
                             find_existing, select_date_split, validate_forecast_shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_glob", default="/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]")
    parser.add_argument("--stage1_prediction_root", required=True)
    parser.add_argument("--output_root", default="./data/stage2_residuals_serd_v1")
    parser.add_argument("--total_error_scaler_path", default="./data/scalers_err_zscore_train.pkl")
    parser.add_argument("--residual_scaler_path", default="./data/scalers_stage2_residual_zscore_train.pkl")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    return parser


def inverse_zscore(value: np.ndarray, scaler_path: str) -> np.ndarray:
    stats = joblib.load(scaler_path)
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    return value.astype(np.float32) * std + mean


class ChannelMoments:
    def __init__(self, channels: int = 5):
        self.count = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.total_sq = np.zeros(channels, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        flat = np.asarray(value, dtype=np.float64).reshape(value.shape[0], -1)
        self.count += flat.shape[1]
        self.total += flat.sum(axis=1)
        self.total_sq += np.square(flat).sum(axis=1)

    def stats(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("No training residuals found; cannot fit the stage-2 scaler")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - mean * mean, 1e-12)
        return {"mean": mean.astype(np.float32), "std": np.sqrt(variance).astype(np.float32)}


def main() -> None:
    args = build_parser().parse_args()
    paper_days = select_date_split(sorted(glob.glob(args.data_root_glob)), "all")
    if not paper_days:
        raise RuntimeError("No day directories in the manuscript period were found")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    moments = ChannelMoments()
    saved = skipped = 0

    for day_text in paper_days:
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        prediction_day = Path(args.stage1_prediction_root) / init_token
        output_day = output_root / day_dir.name
        output_day.mkdir(parents=True, exist_ok=True)
        for lead in LEAD_HOURS:
            forecast_path = find_existing(file_candidates(day_dir, date_token, lead))
            error_path = find_existing(file_candidates(day_dir, date_token, lead, "_err"))
            prediction_path = prediction_day / f"{init_token}_{lead:02d}.npy"
            if forecast_path is None or error_path is None or not prediction_path.is_file():
                skipped += 1
                continue
            forecast = np.load(forecast_path).astype(np.float32)
            validate_forecast_shape(tuple(forecast.shape))
            total_error = np.load(error_path).astype(np.float32)[:, :args.height, :args.width]
            systematic_error = inverse_zscore(np.load(prediction_path), args.total_error_scaler_path)
            systematic_error = systematic_error[:, :args.height, :args.width]
            residual = total_error - systematic_error
            corrected = forecast.copy()
            corrected[list(SURFACE_CHANNEL_INDICES), :args.height, :args.width] += systematic_error
            np.save(output_day / f"{date_token}_{lead:02d}.npy", corrected)
            np.save(output_day / f"{date_token}_{lead:02d}_err.npy", residual.astype(np.float32))
            if day.date().isoformat() <= "2023-09-30":
                moments.update(residual)
            saved += 1

    Path(args.residual_scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(moments.stats(), args.residual_scaler_path)
    print(f"saved_samples={saved} skipped_samples={skipped}")
    print(f"train-only residual scaler: {args.residual_scaler_path}")


if __name__ == "__main__":
    main()
