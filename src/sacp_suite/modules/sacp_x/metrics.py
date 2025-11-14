"""Lightweight chaos metrics for SACP-X."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm


def rosenstein_lle(x: np.ndarray, m: int = 6, tau: int = 4, eps: float = 1e-9) -> float:
    """Estimate the largest Lyapunov exponent using Rosenstein's method."""

    series = np.asarray(x, dtype=float).reshape(-1)
    N = len(series) - (m - 1) * tau
    if N <= 20:
        return 0.0

    embedded = np.stack([series[i : i + N] for i in range(0, m * tau, tau)], axis=1)
    idx = np.arange(N)
    nn = np.zeros(N, dtype=int)

    for i in range(N):
        d = norm(embedded - embedded[i], axis=1)
        exc_start = max(0, i - 10)
        exc_stop = min(N, i + 11)
        d[exc_start:exc_stop] = np.inf
        nn[i] = int(np.argmin(d))

    kmax = min(50, N - 1)
    div = []
    for k in range(1, kmax):
        future_i = idx + k
        future_nn = nn + k
        valid = (future_i < N) & (future_nn < N)
        if not np.any(valid):
            break
        d0 = norm(embedded[valid] - embedded[nn[valid]], axis=1) + eps
        dk = norm(embedded[future_i[valid]] - embedded[future_nn[valid]], axis=1) + eps
        div.append(np.mean(np.log(dk / d0)))

    if not div:
        return 0.0

    t = np.arange(1, len(div) + 1)
    slope = np.polyfit(t, div, 1)[0]
    return float(max(slope, 0.0))
