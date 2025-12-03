from __future__ import annotations

import base64
import io
import os
from typing import List

import numpy as np
import pandas as pd

API_BASE = f"http://{os.getenv('SACP_BIND', '127.0.0.1')}:{os.getenv('SACP_PORT', '8000')}"

# Simple in-process shared trajectory cache (replaces server attribute)
_shared_traj = None


def get_shared_traj():
    return _shared_traj


def set_shared_traj(traj) -> None:
    global _shared_traj
    _shared_traj = traj


def parse_upload(contents: str) -> np.ndarray:
    """Decode base64 CSV upload into a 1D series."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    series = df.iloc[:, 0].astype(float).to_numpy()
    return series


def parse_numeric_list(text: str | None, fallback: List[float]) -> List[float]:
    if text is None:
        return fallback
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    values: List[float] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return fallback
    return values or fallback


def parse_label_list(text: str | None, fallback: List[str]) -> List[str]:
    if text is None:
        return fallback
    values = [p.strip() for p in text.split(",") if p.strip()]
    return values or fallback


def parse_patterns_text(text: str | None, fallback: List[List[float]]) -> List[List[float]]:
    if text is None:
        return fallback
    patterns: List[List[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pattern = [float(tok) for tok in line.replace(" ", "").split(",") if tok]
        except ValueError:
            return fallback
        if pattern:
            patterns.append(pattern)
    return patterns or fallback


def coerce_int(value: float | int | None, default: int, minimum: int = 0) -> int:
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = default
    return max(minimum, val)


def coerce_float(value: float | int | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
