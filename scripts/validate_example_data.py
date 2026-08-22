from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serd.paper.samples import source_sample_manifest


DEFAULT_ROOT = Path("./example_data/CMA_gfs_time_order_3_72")
EXPECTED_LEADS = {
    "20241229": tuple(range(3, 61, 3)),
    "20241230": tuple(range(3, 37, 3)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the bundled two-day example dataset.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def validate_array(path: Path, expected_shape: tuple[int, int, int], expected_dtype: str) -> None:
    try:
        array = np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot read {path} as a NumPy array. If this is an LFS pointer, run `git lfs pull`."
        ) from exc
    if array.shape != expected_shape:
        raise RuntimeError(f"{path}: expected shape {expected_shape}, found {array.shape}")
    if str(array.dtype) != expected_dtype:
        raise RuntimeError(f"{path}: expected dtype {expected_dtype}, found {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{path}: contains NaN or infinite values")


def main() -> None:
    args = parse_args()
    day_dirs = [args.data_root / day for day in EXPECTED_LEADS]
    missing_days = [str(path) for path in day_dirs if not path.is_dir()]
    if missing_days:
        raise RuntimeError(f"Missing example day directories: {missing_days}")

    samples, _ = source_sample_manifest(day_dirs, "test")
    expected_count = sum(len(leads) for leads in EXPECTED_LEADS.values())
    if len(samples) != expected_count:
        raise RuntimeError(f"Expected {expected_count} complete triplets, found {len(samples)}")

    observed = {day: [] for day in EXPECTED_LEADS}
    for sample in samples:
        observed[sample.day_dir.name].append(sample.lead_hour)
        validate_array(sample.forecast_path, (45, 200, 200), "float64")
        validate_array(sample.error_path, (5, 200, 200), "float64")
        validate_array(sample.analysis_path, (5, 200, 200), "float32")

    for day, expected in EXPECTED_LEADS.items():
        actual = tuple(sorted(observed[day]))
        if actual != expected:
            raise RuntimeError(f"{day}: expected leads {expected}, found {actual}")

    print(f"Example dataset OK: {len(samples)} complete triplets across {len(day_dirs)} days")


if __name__ == "__main__":
    main()
