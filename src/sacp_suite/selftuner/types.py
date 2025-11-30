"""Shared datatypes for the self-tuning module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np


@dataclass(frozen=True)
class ChaosBand:
    """Closed interval describing acceptable Lyapunov range."""

    lower: float
    upper: float

    def clamp(self, value: float) -> float:
        return float(min(self.upper, max(self.lower, value)))


@dataclass
class LyapunovEstimate:
    """Streaming Lyapunov spectrum diagnostic."""

    values: np.ndarray
    dt: float
    steps: int
    converged: bool


@dataclass(frozen=True)
class TuningContext:
    """Lightweight metadata describing the controlled system."""

    system_id: str
    modality: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningAction:
    """Control action returned by the self-tuner."""

    spectral_radius_delta: float = 0.0
    gain_delta: float = 0.0
    target_spectral_radius: Optional[float] = None
    target_gain: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "spectral_radius_delta": self.spectral_radius_delta,
            "gain_delta": self.gain_delta,
        }
        if self.target_spectral_radius is not None:
            data["target_spectral_radius"] = self.target_spectral_radius
        if self.target_gain is not None:
            data["target_gain"] = self.target_gain
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data
