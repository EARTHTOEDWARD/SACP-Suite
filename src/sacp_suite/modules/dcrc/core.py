"""Core DCRC components (vendored from Digital Lorenz repo, numpy-only)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

_R_CACHE: dict[int, np.ndarray] = {}


def _get_R(n_features: int = 10) -> np.ndarray:
    """Load/create shared projection matrix (numpy-only to avoid torch heavy dep)."""
    if n_features not in _R_CACHE:
        rng = np.random.default_rng(0)
        _R_CACHE[n_features] = rng.standard_normal((3, n_features))
    return _R_CACHE[n_features]


class SinusoidalMap:
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a, self.b, self.c = a, b, c

    def step(self, s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.stack(
            [
                y,
                np.sin(z),
                self.a + self.b * x + self.c * y - np.sin(z**2),
            ],
            axis=0,
        )

    def iterate(self, x0: np.ndarray, n: int) -> np.ndarray:
        traj = np.empty((3, x0.shape[1], n + 1), dtype=np.float32)
        traj[:, :, 0] = x0
        s = x0
        for t in range(1, n + 1):
            s = self.step(s)
            traj[:, :, t] = s
        return traj


class DCRC:
    def __init__(self, a: float, b: float, c: float, n_iter: int = 60, ridge_alpha: float = 1e-2) -> None:
        self.map = SinusoidalMap(a, b, c)
        self.n_iter = n_iter
        self.readout = Ridge(alpha=ridge_alpha)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        R = _get_R(X.shape[1])
        return np.tanh(X @ R.T).T  # (3, n_samples)

    def _feat(self, X: np.ndarray) -> np.ndarray:
        x0 = self._encode(X)
        traj = self.map.iterate(x0, self.n_iter)
        return traj.transpose(1, 0, 2).reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DCRC":
        self.readout.fit(self._feat(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.readout.predict(self._feat(X))

