"""Streaming Lyapunov estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sacp_suite.selftuner.config import SAFE_EPS
from sacp_suite.selftuner.types import LyapunovEstimate


TangentMapFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class QRStreamingLyapunovEstimator:
    """Implements the classic QR-based streaming Lyapunov spectrum estimator."""

    state_dim: int
    n_exponents: int
    tangent_map: Optional[TangentMapFn]
    window: int = 2000

    def __post_init__(self) -> None:
        if self.n_exponents > self.state_dim:
            raise ValueError("n_exponents cannot exceed state_dim")
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._Q = np.eye(self.state_dim)
        self._sums = np.zeros(self.n_exponents, dtype=float)
        self._steps = 0

    def reset(self) -> None:
        self._reset_accumulators()

    def update(self, state: np.ndarray, dt: float) -> LyapunovEstimate:
        if self.tangent_map is None:
            raise RuntimeError("tangent_map required for Lyapunov estimation")
        jac = self.tangent_map(state, dt)
        if jac.shape != (self.state_dim, self.state_dim):
            raise ValueError("tangent_map returned invalid shape")

        Z = jac @ self._Q
        Q, R = np.linalg.qr(Z)
        self._Q = Q
        diag = np.diag(R)[: self.n_exponents]
        self._sums += np.log(np.abs(diag) + SAFE_EPS)
        self._steps += 1
        converged = self._steps >= self.window
        values = (self._sums / max(self._steps * dt, SAFE_EPS)).copy()
        return LyapunovEstimate(values=values, dt=dt, steps=self._steps, converged=converged)
