"""Reservoir computing core from Fractal LLM Lab (lightweight vendor).

Original: /Users/edward/Desktop/Control Panels/Fractal_LLM_Lab/fractal_llm_lab/src/fractalllm/reservoir.py
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals
from sklearn.linear_model import Ridge


class RCAttractorMem:
    """Reservoir-computing memory for periodic/chaotic attractors."""

    def __init__(
        self,
        n_res: int = 500,
        n_in: int = 4,
        n_out: int = 3,
        leaking_rate: float = 0.2,
        spectral_radius: float = 0.9,
        ridge: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.Win = rng.standard_normal((n_res, n_in)) / np.sqrt(n_in)
        self.Wres = rng.standard_normal((n_res, n_res))
        eigs = eigvals(self.Wres)
        self.Wres *= spectral_radius / max(abs(eigs))
        self.Wout: np.ndarray | None = None
        self.n_out = n_out
        self.leak = leaking_rate
        self.ridge = ridge

    def _step(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single reservoir update."""
        pre = self.Win @ u + self.Wres @ r
        return (1 - self.leak) * r + self.leak * np.tanh(pre)

    def train_batch(self, trajectories: list[np.ndarray], indices: list[float]) -> None:
        """Fit readout weights to match provided attractor trajectories."""
        states, targets = [], []
        for traj, idx in zip(trajectories, indices, strict=False):
            r = np.zeros(self.Wres.shape[0])
            for u in traj:
                r = self._step(r, np.hstack([u, idx]))
                states.append(r.copy())
                targets.append(u)
        X = np.vstack(states)
        Y = np.vstack(targets)
        self.Wout = Ridge(alpha=self.ridge, fit_intercept=False).fit(X, Y).coef_

    def recall(self, idx: float, T: int = 500, r0: np.ndarray | None = None) -> np.ndarray:
        """Closed-loop generation of an attractor."""
        if self.Wout is None:
            raise RuntimeError("Call train_batch before recall.")
        r = np.zeros(self.Wres.shape[0]) if r0 is None else r0.copy()
        out = []
        for _ in range(T):
            v = self.Wout @ r
            r = self._step(r, np.hstack([v, idx]))
            out.append(v)
        return np.array(out)

    def switch(self, r_last: np.ndarray, v_last: np.ndarray, new_idx: float, T: int = 500) -> tuple[bool, np.ndarray]:
        """Attempt in-place switch; return (succ_flag, traj)."""
        if self.Wout is None:
            raise RuntimeError("Call train_batch before switch.")
        r = r_last.copy()
        traj = []
        for _ in range(T):
            r = self._step(r, np.hstack([v_last, new_idx]))
            v_last = self.Wout @ r
            traj.append(v_last)
        return True, np.array(traj)


def generate_reservoir_sample(text: str, n_res: int = 400, coupling: float = 1.0) -> dict:
    """Utility for API/UI: train on synthetic attractors and recall one keyed by text hash."""

    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    t = np.linspace(0, 10, 800)
    base1 = np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)
    base2 = np.stack([np.sin(1.3 * t + 0.8), np.cos(0.7 * t), np.sin(1.7 * t + 1.1)], axis=1)
    idx_val = float(0.15 + 0.7 * (abs(hash(text)) % 10) / 10.0)

    rc = RCAttractorMem(
        n_res=n_res,
        n_in=4,
        n_out=3,
        spectral_radius=max(0.1, min(1.5, coupling)),
        leaking_rate=0.2,
        ridge=1e-5,
        seed=42,
    )
    rc.train_batch([base1, base2], [0.2, 0.8])
    traj = rc.recall(idx_val, T=400)
    r0 = np.zeros(rc.Wres.shape[0])
    r0[:3] = traj[-1] if len(traj) else 0.0
    switched_ok, switched = rc.switch(r0, traj[-1], 1.0 - idx_val, T=200)
    return {
        "trajectory": traj.tolist(),
        "switch": switched.tolist() if switched_ok else [],
        "index": idx_val,
    }

