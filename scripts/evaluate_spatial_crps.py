"""Compute Figure 8 mean t2m CRPS maps for SERD, CorrDiff, and their difference."""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np

from serd.paper.metrics import empirical_crps
from serd.paper.spec import file_candidates, find_existing, select_date_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serd_root", default="./outputs/predictions/serd_v1/stage2_serd")
    parser.add_argument("--corrdiff_root", default="./outputs/predictions/corrdiff")
    parser.add_argument("--target_root_glob", default="/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]")
    parser.add_argument("--out_path", default="./outputs/metrics/figure8_t2m_spatial_crps.npz")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=192)
    args = parser.parse_args()
    leads = (24, 48, 72)
    sums = {method: np.zeros((3, args.height, args.width), dtype=np.float64) for method in ("serd", "corrdiff")}
    counts = np.zeros(3, dtype=np.int64)
    roots = {"serd": Path(args.serd_root), "corrdiff": Path(args.corrdiff_root)}
    for day_text in select_date_split(sorted(glob.glob(args.target_root_glob)), "test"):
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        for lead_index, lead in enumerate(leads):
            analysis_path = find_existing(file_candidates(day_dir, date_token, lead, "_analysis"))
            sample_paths = {method: root / init_token / f"{init_token}_{lead:02d}.npy" for method, root in roots.items()}
            if analysis_path is None or any(not path.is_file() for path in sample_paths.values()):
                continue
            observation = np.load(analysis_path).astype(np.float32)[4, :args.height, :args.width]
            for method, path in sample_paths.items():
                ensemble = np.load(path).astype(np.float32)[:, 4, :args.height, :args.width]
                if ensemble.shape[0] != 16:
                    raise ValueError(f"Figure 8 requires K=16: {path}")
                sums[method][lead_index] += empirical_crps(ensemble, observation)
            counts[lead_index] += 1
    if np.any(counts == 0):
        raise RuntimeError(f"No matched test samples for one or more Figure 8 leads: counts={counts.tolist()}")
    serd = sums["serd"] / counts[:, None, None]
    corrdiff = sums["corrdiff"] / counts[:, None, None]
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, leads=np.asarray(leads), serd_crps=serd.astype(np.float32),
             corrdiff_crps=corrdiff.astype(np.float32), difference=(serd - corrdiff).astype(np.float32), counts=counts)
    print(out_path)


if __name__ == "__main__":
    main()
