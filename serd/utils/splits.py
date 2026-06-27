from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SplitSpec:
    train_count: int = 1292
    valid_count: int = 92

    @property
    def valid_start(self) -> int:
        return self.train_count

    @property
    def test_start(self) -> int:
        return self.train_count + self.valid_count


def select_split(paths: Sequence[str], split: str, spec: SplitSpec) -> list[str]:
    paths = list(paths)
    if split == "train":
        return paths[: spec.train_count]
    if split == "valid":
        return paths[spec.valid_start: spec.test_start]
    if split == "test":
        return paths[spec.test_start:]
    if split == "all":
        return paths
    raise ValueError(f"Unknown split: {split}")
