"""Require every probabilistic method to cover the identical paper test set."""
from __future__ import annotations

import argparse
import glob

from serd.paper.samples import (
    require_common_method_outputs,
    require_expected_source_count,
    source_sample_manifest,
)


DEFAULT_METHOD_ROOTS = {
    "GridLeadBias": "./outputs/predictions/gridleadbias",
    "NGR-like Gaussian MOS": "./outputs/predictions/ngr_like_gaussian_mos",
    "CorrDiff": "./outputs/predictions/corrdiff",
    "Direct diffusion + fCRPS": "./outputs/predictions/direct_diffusion_fcrps",
    "Two-stage w/o fCRPS": "./outputs/predictions/twostage_no_fcrps",
    "SERD": "./outputs/predictions/serd_v1/stage2_serd",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_root_glob",
        default="./example_data/CMA_gfs_time_order_3_72/*[0-9]",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument(
        "--method_root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override defaults with one or more method output roots.",
    )
    args = parser.parse_args()
    method_roots = DEFAULT_METHOD_ROOTS.copy()
    for value in args.method_root:
        if "=" not in value:
            raise ValueError(f"--method_root must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        method_roots[name] = path
    samples, excluded = source_sample_manifest(
        sorted(glob.glob(args.target_root_glob)), args.split
    )
    require_expected_source_count(samples, args.split)
    require_common_method_outputs(samples, method_roots)
    print(
        f"strict_common_samples=ok split={args.split} included={len(samples)} "
        f"source_excluded={len(excluded)} methods={len(method_roots)}"
    )


if __name__ == "__main__":
    main()
