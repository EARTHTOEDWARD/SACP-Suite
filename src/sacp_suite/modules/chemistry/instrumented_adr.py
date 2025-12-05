"""Lightweight Autocatalytic Duffing Ring helper with logging and CLC proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass
class ADRLog:
    time: np.ndarray
    x: np.ndarray
    v: np.ndarray
    r: np.ndarray
    k: np.ndarray
    R_global: np.ndarray
    R_local: np.ndarray


def _normalized_entropy(x: np.ndarray, bins: int = 64) -> float:
    """Return entropy(x)/log(bins) as a simple complexity proxy."""
    hist, _ = np.histogram(x, bins=bins, density=False)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(len(p)))


def _decorrelation_time(x: np.ndarray, dt: float) -> float:
    """Rough decorrelation time: first lag where autocorr < 1/e."""
    x_centered = x - np.mean(x)
    ac = np.correlate(x_centered, x_centered, mode="full")
    ac = ac[ac.size // 2 :]
    if ac[0] == 0:
        return 0.0
    ac = ac / ac[0]
    below = np.nonzero(ac < np.exp(-1))[0]
    lag = int(below[0]) if below.size else len(ac) - 1
    return float(lag * dt)


def compute_clc_proxy(log: Dict[str, np.ndarray], dt: float) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
    """
    Compute CLC-style diagnostics from an ADR log.

    Contract (Metabolism Atlas compatible):
      - Input: log dict with x (T,N) at minimum; dt in seconds.
      - Output: S_node array-like (len N), S_ring scalar, diag dict of array-like.
    """
    X = np.asarray(log["x"])
    N = X.shape[1]

    entropies = np.array([_normalized_entropy(X[:, i]) for i in range(N)], dtype=float)
    taus = np.array([_decorrelation_time(X[:, i], dt) for i in range(N)], dtype=float)

    # Spatial reach: mean absolute correlation to other sites (per-site).
    if N > 1:
        corr = np.corrcoef(X.T)
        R_spatial = np.zeros(N, dtype=float)
        for i in range(N):
            others = np.delete(np.abs(corr[i]), i)
            R_spatial[i] = float(np.nanmean(others)) if others.size else 0.0
    else:
        R_spatial = np.zeros(N, dtype=float)

    C_inf = entropies  # use entropy as a capture proxy per site
    tau_past = taus
    tau_future = taus

    S_node = C_inf * np.sqrt(tau_past * tau_future) * np.maximum(R_spatial, 1e-8)
    S_ring = float(np.mean(S_node))
    diag = {
        "C_inf": C_inf,
        "tau_past": tau_past,
        "tau_future": tau_future,
        "R_spatial": R_spatial,
    }
    return S_node, S_ring, diag


class InstrumentedADR:
    """Minimal ADR integrator with logging suitable for dashboards."""

    def __init__(
        self,
        N: int,
        dt: float,
        a: np.ndarray,
        gamma: np.ndarray,
        mu: float,
        sigma: float,
        eta: float,
        F: float = 0.0,
        omega: float = 1.0,
        k_init: float = 0.3,
        k_min: float = 0.0,
        k_max: float = 1.5,
        plasticity_enabled: bool = False,
        eta_plast: float = 1e-3,
        R_target: float = 0.6,
        plasticity_every: int = 10,
        neighborhood_radius: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.N = int(N)
        self.dt = float(dt)
        self.a = np.asarray(a, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.F = float(F)
        self.omega = float(omega)
        self.k = np.full(self.N, float(k_init), dtype=float)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.plasticity_enabled = bool(plasticity_enabled)
        self.eta_plast = float(eta_plast)
        self.R_target = float(R_target)
        self.plasticity_every = int(max(plasticity_every, 1))
        self.neighborhood_radius = int(max(neighborhood_radius, 1))

        rng = np.random.default_rng(seed)
        self.x = rng.uniform(-1.0, 1.0, size=self.N)
        self.v = rng.uniform(-0.2, 0.2, size=self.N)
        self.r = np.zeros(self.N, dtype=float)
        self.t = 0.0
        self._step_count = 0

    def _laplacian(self, x: np.ndarray) -> np.ndarray:
        return np.roll(x, -1) + np.roll(x, 1) - 2.0 * x

    def _update_plasticity(self) -> None:
        """Toy Hebbian plasticity on the diffusive coupling."""
        if not self.plasticity_enabled:
            return
        if self._step_count % self.plasticity_every != 0:
            return
        local_drive = np.mean(np.abs(self.r))
        delta = self.eta_plast * (local_drive - self.R_target)
        self.k = np.clip(self.k + delta, self.k_min, self.k_max)

    def step(self) -> None:
        dt = self.dt
        lap = self._laplacian(self.x)
        r_prev = np.roll(self.r, 1)

        accel = (
            -self.gamma * self.v
            + self.a * self.x
            - self.x**3
            + self.k * lap
            + self.eta * r_prev
            + self.F * np.cos(self.omega * self.t)
        )

        self.v = self.v + dt * accel
        self.x = self.x + dt * self.v
        self.r = self.r + dt * (-self.mu * self.r + self.sigma * self.x**2)

        self.t += dt
        self._step_count += 1
        self._update_plasticity()

    def run(self, n_steps: int, log_every: int = 1) -> Dict[str, np.ndarray]:
        """Integrate forward and return time series logs."""
        log_every = max(1, int(log_every))
        times = []
        xs = []
        vs = []
        rs = []
        ks = []
        R_global = []
        R_local = []

        for step in range(int(n_steps)):
            if step % log_every == 0:
                times.append(self.t)
                xs.append(self.x.copy())
                vs.append(self.v.copy())
                rs.append(self.r.copy())
                ks.append(self.k.copy())
                R_global.append(np.mean(self.r))
                R_local.append(self.r.copy())
            self.step()

        return {
            "time": np.array(times, dtype=float),
            "x": np.array(xs, dtype=float),
            "v": np.array(vs, dtype=float),
            "r": np.array(rs, dtype=float),
            "k": np.array(ks, dtype=float),
            "R_global": np.array(R_global, dtype=float),
            "R_local": np.array(R_local, dtype=float),
        }
