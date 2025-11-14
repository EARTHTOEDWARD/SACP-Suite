"""Attractor-Based Trajectory Calculator (ABTC) stubs."""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4(f: Callable[[np.ndarray], np.ndarray], x0, dt: float, n: int) -> np.ndarray:
    """Generic RK4 helper used by trajectory solvers."""

    x = np.array(x0, dtype=float).copy()
    traj = [x.copy()]
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.copy())
    return np.array(traj)
