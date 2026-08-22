"""Build CorrDiff target residuals and their train-only scaler.

The deterministic stage predicts analysis directly.  This script stores its
physical prediction in the five surface slots of the 45-channel conditioning
forecast, and stores analysis-minus-prediction as the diffusion target.  The
generic stage-2 inference code can therefore reconstruct final fields by adding
sampled residuals to those surface slots.
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from serd.paper.spec import LEAD_HOURS, SURFACE_CHANNEL_INDICES, file_candidates, find_existing, select_date_split, validate_forecast_shape


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data_root_glob", default="./example_data/CMA_gfs_time_order_3_72/*[0-9]")
    value.add_argument("--stage1_prediction_root", required=True)
    value.add_argument("--analysis_scaler_path", default="./data/scalers_ana_zscore_two_step_unet_train.pkl")
    value.add_argument("--output_root", default="./data/corrdiff_residuals")
    value.add_argument("--residual_scaler_path", default="./data/scalers_corrdiff_residual_zscore_train.pkl")
    value.add_argument("--height", type=int, default=192)
    value.add_argument("--width", type=int, default=192)
    return value


def main() -> None:
    args = parser().parse_args()
    analysis_stats = joblib.load(args.analysis_scaler_path)
    mean = np.asarray(analysis_stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(analysis_stats["std"], dtype=np.float32)[:, None, None]
    sums = np.zeros(5, dtype=np.float64)
    squares = np.zeros(5, dtype=np.float64)
    count = saved = 0
    for day_text in select_date_split(sorted(glob.glob(args.data_root_glob)), "all"):
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        output_day = Path(args.output_root) / day_dir.name
        output_day.mkdir(parents=True, exist_ok=True)
        for lead in LEAD_HOURS:
            forecast_path = find_existing(file_candidates(day_dir, date_token, lead))
            analysis_path = find_existing(file_candidates(day_dir, date_token, lead, "_analysis"))
            prediction_path = Path(args.stage1_prediction_root) / init_token / f"{init_token}_{lead:02d}.npy"
            if forecast_path is None or analysis_path is None or not prediction_path.is_file():
                continue
            forecast = np.load(forecast_path).astype(np.float32)
            validate_forecast_shape(tuple(forecast.shape))
            analysis = np.load(analysis_path).astype(np.float32)[:, :args.height, :args.width]
            prediction = np.load(prediction_path).astype(np.float32) * std + mean
            prediction = prediction[:, :args.height, :args.width]
            residual = analysis - prediction
            conditioning = forecast.copy()
            conditioning[list(SURFACE_CHANNEL_INDICES), :args.height, :args.width] = prediction
            np.save(output_day / f"{date_token}_{lead:02d}.npy", conditioning)
            np.save(output_day / f"{date_token}_{lead:02d}_err.npy", residual)
            if day.date().isoformat() <= "2023-09-30":
                flat = residual.astype(np.float64).reshape(5, -1)
                sums += flat.sum(axis=1)
                squares += np.square(flat).sum(axis=1)
                count += flat.shape[1]
            saved += 1
    if count == 0:
        raise RuntimeError("No CorrDiff training residuals found")
    residual_mean = sums / count
    residual_std = np.sqrt(np.maximum(squares / count - residual_mean ** 2, 1e-12))
    Path(args.residual_scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"mean": residual_mean.astype(np.float32), "std": residual_std.astype(np.float32)}, args.residual_scaler_path)
    print(f"saved_samples={saved} residual_scaler={args.residual_scaler_path}")


if __name__ == "__main__":
    main()
