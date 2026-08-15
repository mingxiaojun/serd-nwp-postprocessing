from dataclasses import dataclass
from datetime import date
from typing import Sequence

from serd.paper.spec import PAPER_SPEC, select_date_split


@dataclass(frozen=True)
class SplitSpec:
    train_start: date = PAPER_SPEC.train_start
    train_end: date = PAPER_SPEC.train_end
    valid_start: date = PAPER_SPEC.valid_start
    valid_end: date = PAPER_SPEC.valid_end
    test_start: date = PAPER_SPEC.test_start
    test_end: date = PAPER_SPEC.test_end

def select_split(paths: Sequence[str], split: str, spec: SplitSpec) -> list[str]:
    del spec
    return select_date_split(paths, split)
