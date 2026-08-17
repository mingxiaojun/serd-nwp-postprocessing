"""Compute Figure 8 mean t2m CRPS maps for SERD, CorrDiff, and their difference."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from serd.paper.metrics import empirical_crps
from serd.paper.samples import (require_common_method_outputs,
                                require_expected_source_count,
                                source_sample_manifest)


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
    samples, _ = source_sample_manifest(sorted(glob.glob(args.target_root_glob)), "test")
    require_expected_source_count(samples, "test")
    figure_samples = [sample for sample in samples if sample.lead_hour in leads]
    require_common_method_outputs(figure_samples, roots)
    for sample in figure_samples:
        lead_index = leads.index(sample.lead_hour)
        observation = np.load(sample.analysis_path).astype(np.float32)[4, :args.height, :args.width]
        for method, root in roots.items():
            path = sample.ensemble_path(root)
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
             corrdiff_crps=corrdiff.astype(np.float32), difference=(corrdiff - serd).astype(np.float32), counts=counts)
    print(out_path)


if __name__ == "__main__":
    main()
