from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


SURFACE_VARIABLES = ("q2m", "u10", "v10", "sp", "t2m")
PRESSURE_LEVELS_HPA = (925, 850, 700, 500, 300, 200, 150, 100)
LEAD_HOURS = tuple(range(3, 73, 3))
FORECAST_CHANNELS = tuple(
    channel
    for surface, upper in zip(SURFACE_VARIABLES, ("q", "u", "v", "z", "t"))
    for channel in (surface, *(f"{upper}{level}" for level in PRESSURE_LEVELS_HPA))
)
SURFACE_CHANNEL_INDICES = (0, 9, 18, 27, 36)


@dataclass(frozen=True)
class PaperSpec:
    data_root: str = "./example_data/CMA_gfs_time_order_3_72"
    train_start: date = date(2020, 1, 1)
    train_end: date = date(2023, 9, 30)
    valid_start: date = date(2023, 10, 1)
    valid_end: date = date(2023, 12, 31)
    test_start: date = date(2024, 1, 1)
    test_end: date = date(2024, 12, 31)
    height: int = 192
    width: int = 192
    init_hour_utc: int = 9
    sigma_min: float = 0.001
    sigma_max: float = 10.0
    train_members: int = 8
    inference_members: int = 16
    learning_rate: float = 2e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6

    def split_for(self, value: date | datetime) -> str | None:
        day = value.date() if isinstance(value, datetime) else value
        if self.train_start <= day <= self.train_end:
            return "train"
        if self.valid_start <= day <= self.valid_end:
            return "valid"
        if self.test_start <= day <= self.test_end:
            return "test"
        return None


PAPER_SPEC = PaperSpec()


def day_from_path(path: str | Path) -> date:
    token = Path(path).name
    for fmt in ("%Y%m%d", "%Y_%m_%d", "%Y-%m-%d-%H"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse initialization date from {path!s}")


def select_date_split(paths: Iterable[str | Path], split: str) -> list[str]:
    if split not in {"train", "valid", "test", "all"}:
        raise ValueError(f"Unknown split: {split}")
    ordered = sorted(str(path) for path in paths)
    if split == "all":
        return [path for path in ordered if PAPER_SPEC.split_for(day_from_path(path))]
    return [path for path in ordered if PAPER_SPEC.split_for(day_from_path(path)) == split]


def forecast_filename(date_token: str, lead_hour: int) -> str:
    return f"{date_token}_{lead_hour:02d}.npy"


def error_filename(date_token: str, lead_hour: int) -> str:
    return f"{date_token}_{lead_hour:02d}_err.npy"


def file_candidates(day_dir: str | Path, date_token: str, lead_hour: int, suffix: str = "") -> tuple[Path, ...]:
    root = Path(day_dir)
    return (
        root / f"{date_token}_{lead_hour}{suffix}.npy",
        root / f"{date_token}_{lead_hour:02d}{suffix}.npy",
        root / f"{date_token}__{lead_hour}{suffix}.npy",
        root / f"{date_token}__{lead_hour:02d}{suffix}.npy",
    )


def find_existing(candidates: Iterable[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def validate_forecast_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 3 or shape[0] != 45:
        raise ValueError(f"Expected forecast [45,H,W] in paper channel order, got {shape}")


def split_forecast_channels(array):
    """Return paper inputs: five surface fields and 5x8 upper-air fields."""
    validate_forecast_shape(tuple(array.shape))
    surface = array[list(SURFACE_CHANNEL_INDICES)]
    upper = array.reshape(5, 9, *array.shape[-2:])[:, 1:]
    return surface, upper
