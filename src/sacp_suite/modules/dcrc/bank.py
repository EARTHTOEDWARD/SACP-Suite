"""DCRC bank (multiple maps) vendor."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .core import DCRC


class DCRC_Bank:
    """Concatenates features from N independent sinusoidal maps, then fits one read-out."""

    def __init__(self, n_maps: int = 3, n_iter: int = 60, ridge_alpha: float = 1e-2) -> None:
        self.maps = []
        for _ in range(n_maps):
            dcrc = DCRC(
                a=np.random.uniform(-0.08, 0.40),
                b=np.random.uniform(0.75, 0.90),
                c=np.random.uniform(0.96, 1.04),
                n_iter=n_iter,
                ridge_alpha=ridge_alpha,
            )
            dcrc._R = np.random.randn(3, 10)  # type: ignore[attr-defined]
            self.maps.append(dcrc)
        self.readout = Ridge(alpha=ridge_alpha)

    def _get_features_for_map(self, m: DCRC, X: np.ndarray) -> np.ndarray:
        if hasattr(m, "_R"):
            R_custom = m._R  # type: ignore[attr-defined]
            x0 = np.tanh(X @ R_custom.T).T
            traj = m.map.iterate(x0, m.n_iter)
            return traj.transpose(1, 0, 2).reshape(X.shape[0], -1)
        return m._feat(X)

    def _feat_bank(self, X: np.ndarray) -> np.ndarray:
        feats = [self._get_features_for_map(m, X) for m in self.maps]
        return np.concatenate(feats, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DCRC_Bank":
        for m in self.maps:
            _ = self._get_features_for_map(m, X[:1])
        self.readout.fit(self._feat_bank(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.readout.predict(self._feat_bank(X))


# alias for plugin API compatibility
DCRCBank = DCRC_Bank

