from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_feature_base(path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one processed classification base.

    The last column is treated as the binary handover label. This keeps the
    loader compatible with the original ANATEL concatbase files whose final
    column is sometimes named `50` rather than `label`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset must contain features and one label column: {path}")

    label_col = df.columns[-1]
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    y = df[label_col].round().astype(int).to_numpy()

    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1 in {path}; got {sorted(unique)}")

    return X, y, df


def dataset_summary(y: np.ndarray) -> dict[str, float | int]:
    """Return basic label-distribution statistics."""
    n = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return {
        "samples": n,
        "negatives": negatives,
        "positives": positives,
        "positive_ratio": positives / n if n else 0.0,
    }
