"""Regenerate the paper-period inclusion/exclusion manifest from source arrays."""
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

import numpy as np

from serd.paper.spec import LEAD_HOURS, file_candidates, find_existing, select_date_split, validate_forecast_shape
from serd.paper.samples import EXPECTED_POST_QC_SAMPLES


EXPECTED_COUNTS = EXPECTED_POST_QC_SAMPLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_glob", default="/online1/linxin_group/wangmingming/data/CMA_gfs_time_order_3_72/*[0-9]")
    parser.add_argument("--out_dir", default="./outputs/manifests")
    parser.add_argument("--strict", action="store_true", help="Fail when regenerated counts differ from the manuscript")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_days = sorted(glob.glob(args.data_root_glob))
    summary = []
    for split, expected in EXPECTED_COUNTS.items():
        included = []
        excluded = []
        for day_text in select_date_split(all_days, split):
            day_dir = Path(day_text)
            date_token = datetime.strptime(day_dir.name, "%Y%m%d").strftime("%Y_%m_%d")
            for lead in LEAD_HOURS:
                paths = {
                    "forecast": find_existing(file_candidates(day_dir, date_token, lead)),
                    "error": find_existing(file_candidates(day_dir, date_token, lead, "_err")),
                    "analysis": find_existing(file_candidates(day_dir, date_token, lead, "_analysis")),
                }
                reason = ""
                if any(path is None for path in paths.values()):
                    reason = "missing:" + ",".join(name for name, path in paths.items() if path is None)
                else:
                    try:
                        forecast = np.load(paths["forecast"], mmap_mode="r")
                        error = np.load(paths["error"], mmap_mode="r")
                        analysis = np.load(paths["analysis"], mmap_mode="r")
                        validate_forecast_shape(tuple(forecast.shape))
                        if tuple(error.shape[:1]) != (5,) or tuple(analysis.shape[:1]) != (5,):
                            reason = f"shape:forecast={forecast.shape};error={error.shape};analysis={analysis.shape}"
                    except (OSError, ValueError) as exc:
                        reason = f"unreadable:{exc}"
                row = {"date": day_dir.name, "lead_hour": lead, "forecast": paths["forecast"] or "",
                       "error": paths["error"] or "", "analysis": paths["analysis"] or "", "reason": reason}
                (excluded if reason else included).append(row)
        fields = ("date", "lead_hour", "forecast", "error", "analysis", "reason")
        for label, rows in (("included", included), ("excluded", excluded)):
            with (out_dir / f"{split}_{label}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        summary.append((split, len(included), len(excluded), expected))
    for split, included, excluded, expected in summary:
        print(f"{split}: included={included} excluded={excluded} manuscript_expected={expected}")
    mismatches = [item for item in summary if item[1] != item[3]]
    if args.strict and mismatches:
        raise RuntimeError(f"Post-QC sample counts differ from manuscript: {mismatches}")


if __name__ == "__main__":
    main()
