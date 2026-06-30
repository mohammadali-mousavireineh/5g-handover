from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one processed feature dataset."""

    feature_model: str
    filename: str


DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("LSTM", "anatel_concatbases.csv"),
    DatasetSpec("GRU", "anatel_concatbases_gru.csv"),
    DatasetSpec("TCN", "anatel_concatbases_tcn.csv"),
)


def project_root() -> Path:
    """Return repository root when running from the source tree."""
    return Path(__file__).resolve().parents[2]
