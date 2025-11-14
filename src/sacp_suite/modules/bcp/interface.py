"""Bioelectric Control Panel (BCP) data interface stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Recording:
    """Simple container representing an acquired recording."""

    id: str
    t: np.ndarray
    channels: List[str]
    values: np.ndarray  # shape (N, C)


def to_section(points: np.ndarray) -> np.ndarray:
    """Placeholder for Poincaré or fixed-cadence sectioning."""

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected (N,2) section points")
    return arr[:, :2]
