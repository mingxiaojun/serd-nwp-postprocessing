"""Generate the 16-member physical GridLeadBias ensemble used for comparison."""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np

from serd.paper.spec import LEAD_HOURS, SURFACE_CHANNEL_INDICES, file_candidates, find_existing, select_date_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params_path", default="./outputs/checkpoints/gridleadbias/allvars_gridlead_bias_params.npz")
    parser.add_argument("--data_root_glob", default="/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]")
    parser.add_argument("--output_root", default="./outputs/predictions/gridleadbias")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--ensemble_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    args = parser.parse_args()
    if args.ensemble_size != 16:
        raise ValueError("Paper comparison requires 16 members")
    params = np.load(args.params_path)
    lead_values = params["lead_values"].astype(int)
    rng = np.random.default_rng(args.seed)
    saved = 0
    for day_text in select_date_split(sorted(glob.glob(args.data_root_glob)), args.split):
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        output_day = Path(args.output_root) / init_token
        output_day.mkdir(parents=True, exist_ok=True)
        for lead in LEAD_HOURS:
            forecast_path = find_existing(file_candidates(day_dir, date_token, lead))
            matches = np.flatnonzero(lead_values == (lead // 3 - 1))
            if forecast_path is None or not matches.size:
                continue
            li = int(matches[0])
            forecast = np.load(forecast_path).astype(np.float32)
            surface = forecast[list(SURFACE_CHANNEL_INDICES), :args.height, :args.width]
            bias = params["bias"][:, li, :args.height, :args.width]
            sigma = params["sigma"][:, li, :args.height, :args.width]
            noise = rng.standard_normal((16, *surface.shape)).astype(np.float32)
            ensemble = surface[None] + bias[None] + sigma[None] * noise
            np.save(output_day / f"{init_token}_{lead:02d}.npy", ensemble.astype(np.float32))
            saved += 1
    print(f"saved_samples={saved}")


if __name__ == "__main__":
    main()
