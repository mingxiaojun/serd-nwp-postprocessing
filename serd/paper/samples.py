from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from .spec import LEAD_HOURS, file_candidates, find_existing, select_date_split


EXPECTED_POST_QC_SAMPLES = {"train": 32560, "valid": 2208, "test": 8685}


def ensemble_filename(init_token: str, lead_hour: int) -> str:
    """Canonical paper output name, including zero padding for 3/6/9 h."""
    return f"{init_token}_{lead_hour:02d}.npy"


@dataclass(frozen=True)
class PaperSample:
    day_dir: Path
    date_token: str
    init_token: str
    lead_hour: int
    forecast_path: Path
    error_path: Path
    analysis_path: Path

    @property
    def key(self) -> str:
        return f"{self.init_token}/{ensemble_filename(self.init_token, self.lead_hour)}"

    def ensemble_path(self, root: str | Path) -> Path:
        return Path(root) / self.init_token / ensemble_filename(self.init_token, self.lead_hour)


def source_sample_manifest(
    day_paths: Iterable[str | Path], split: str
) -> tuple[list[PaperSample], dict[str, tuple[str, ...]]]:
    """Build the one canonical source-sample set shared by every paper method."""
    included: list[PaperSample] = []
    excluded: dict[str, tuple[str, ...]] = {}
    for day_text in select_date_split(day_paths, split):
        day_dir = Path(day_text)
        day = datetime.strptime(day_dir.name, "%Y%m%d")
        date_token = day.strftime("%Y_%m_%d")
        init_token = day.replace(hour=9).strftime("%Y-%m-%d-%H")
        for lead in LEAD_HOURS:
            paths = {
                "forecast": find_existing(file_candidates(day_dir, date_token, lead)),
                "error": find_existing(file_candidates(day_dir, date_token, lead, "_err")),
                "analysis": find_existing(file_candidates(day_dir, date_token, lead, "_analysis")),
            }
            key = f"{init_token}/{ensemble_filename(init_token, lead)}"
            missing = tuple(name for name, path in paths.items() if path is None)
            if missing:
                excluded[key] = missing
                continue
            included.append(PaperSample(
                day_dir=day_dir,
                date_token=date_token,
                init_token=init_token,
                lead_hour=lead,
                forecast_path=paths["forecast"],
                error_path=paths["error"],
                analysis_path=paths["analysis"],
            ))
    return included, excluded


def require_expected_source_count(samples: Iterable[PaperSample], split: str) -> None:
    expected = EXPECTED_POST_QC_SAMPLES[split]
    actual = len(samples) if hasattr(samples, "__len__") else sum(1 for _ in samples)
    if actual != expected:
        raise RuntimeError(
            f"Strict paper sample check failed for {split}: expected {expected}, found {actual}. "
            "Regenerate the inclusion/exclusion manifest before evaluation."
        )


def missing_method_outputs(
    samples: Iterable[PaperSample], method_roots: Mapping[str, str | Path]
) -> dict[str, list[str]]:
    sample_list = list(samples)
    return {
        method: [sample.key for sample in sample_list if not sample.ensemble_path(root).is_file()]
        for method, root in method_roots.items()
    }


def require_common_method_outputs(
    samples: Iterable[PaperSample], method_roots: Mapping[str, str | Path]
) -> None:
    missing = {method: keys for method, keys in missing_method_outputs(samples, method_roots).items() if keys}
    if missing:
        summary = "; ".join(
            f"{method}: missing={len(keys)}, first={keys[:3]}" for method, keys in missing.items()
        )
        raise RuntimeError(f"Strict common-sample check failed: {summary}")
