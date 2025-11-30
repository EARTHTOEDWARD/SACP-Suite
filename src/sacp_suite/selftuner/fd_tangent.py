"""Finite-difference helpers for tangent map construction."""

from __future__ import annotations

from typing import Callable

import numpy as np

from sacp_suite.selftuner.config import JACOBIAN_EPS
from sacp_suite.selftuner.lyapunov import TangentMapFn


PureStepFn = Callable[[np.ndarray, float], np.ndarray]


def make_fd_tangent_map_from_step(step_fn: PureStepFn, eps: float = JACOBIAN_EPS) -> TangentMapFn:
    """Create a tangent map from a pure (side-effect-free) step function."""

    def tangent_map(x: np.ndarray, dt: float) -> np.ndarray:
        base = step_fn(x, dt)
        n = x.size
        jac = np.zeros((n, n), dtype=float)
        for i in range(n):
            perturb = np.zeros_like(x)
            perturb[i] = eps
            fx = step_fn(x + perturb, dt)
            jac[:, i] = (fx - base) / eps
        return jac

    return tangent_map
