"""MetaAttractor SPSA loop orchestrating ABTC → Attractorhedron → Fractalhedron.

This module packages a simple SPSA-based self-tuner over (rho, alpha):
  - rho: Lorenz-63 control parameter for the plant
  - alpha: gate factor applied to Attractorhedron's left/right operator split

Each meta-step:
  1) evaluate J at v_plus, v_minus via episodes + operators + multifractal stats
  2) estimate gradient with SPSA
  3) update v and project into a safe box

The implementation keeps all dependencies within the Suite (numpy + existing
Attractorhedron/Fractalhedron helpers). ABTC-style episode generation is
implemented here for Lorenz-63 with a simple Poincaré section sampler; swap
`run_episode` if you have a richer engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from sacp_suite.modules.attractorhedron.operator import analyze_operator, build_ulam, left_right_gate
from sacp_suite.modules.fractalhedron.core import build_fractalhedron_k, build_symbolic_sequence


# ---------------------------------------------------------------------------
# SPSA configuration and state
# ---------------------------------------------------------------------------


@dataclass
class SPSAConfig:
    a0: float = 0.2
    c0: float = 0.05
    A: float = 10.0
    alpha: float = 0.602
    gamma: float = 0.101
    precond: np.ndarray = field(default_factory=lambda: np.array([0.3, 1.0], dtype=float))
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((20.0, 40.0), (0.1, 1.0))

    def gains(self, k: int) -> Tuple[float, float]:
        a_k = self.a0 / ((k + self.A) ** self.alpha)
        c_k = self.c0 / ((k + 1) ** self.gamma)
        return float(a_k), float(c_k)


@dataclass
class MetaState:
    k: int
    v: np.ndarray  # [rho, alpha]
    history: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Episode: Lorenz-63 + Poincaré section sampler
# ---------------------------------------------------------------------------


def lorenz63_deriv(x: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    px, py, pz = x
    dx = sigma * (py - px)
    dy = px * (rho - pz) - py
    dz = px * py - beta * pz
    return np.array([dx, dy, dz], dtype=float)


def run_lorenz_episode(
    rho: float,
    target_crossings: int = 8000,
    dt: float = 0.01,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    seed: int | None = None,
    max_steps: int = 200000,
) -> Dict[str, object]:
    """Integrate Lorenz-63 and collect Poincaré section hits on y=0, ẏ>0."""
    rng = np.random.default_rng(seed)
    x = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)
    hits: List[Tuple[float, float]] = []
    energy_acc = 0.0
    steps = 0
    prev_y = x[1]

    while len(hits) < target_crossings and steps < max_steps:
        k1 = lorenz63_deriv(x, sigma, rho, beta)
        k2 = lorenz63_deriv(x + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz63_deriv(x + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz63_deriv(x + dt * k3, sigma, rho, beta)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        cur_y = x[1]
        if prev_y < 0.0 <= cur_y:
            hits.append((x[0], x[2]))
        prev_y = cur_y
        energy_acc += float(np.dot(k1, k1))
        steps += 1

    section_hits = np.array(hits, dtype=float) if hits else np.zeros((0, 2))
    energy = energy_acc / max(steps, 1)
    return {"section_hits": section_hits, "raw_metrics": {"energy": energy, "steps": steps}}


# ---------------------------------------------------------------------------
# Cost assembly
# ---------------------------------------------------------------------------


def compute_cost(
    lam2_abs: float,
    gamma: float,
    D2: float,
    energy: float,
    *,
    gamma_target: float = 0.05,
    D2_target: float = 1.0,
    w_lambda: float = 1.0,
    w_gamma: float = 0.5,
    w_D2: float = 0.5,
    w_energy: float = 0.05,
) -> float:
    """Simple geometry + energy cost."""
    pen_gamma = max(0.0, gamma_target - gamma)
    pen_D2 = max(0.0, D2_target - D2)
    return w_lambda * lam2_abs + w_gamma * pen_gamma + w_D2 * pen_D2 + w_energy * energy


# ---------------------------------------------------------------------------
# MetaAttractor class
# ---------------------------------------------------------------------------


class MetaAttractor:
    """SPSA-based meta-optimizer over (rho, alpha) with Attractorhedron/Fractalhedron metrics."""

    def __init__(
        self,
        spsa_cfg: SPSAConfig | None = None,
        run_episode: Callable[[float], Dict[str, object]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spsa_cfg = spsa_cfg or SPSAConfig()
        self.run_episode = run_episode or (lambda rho: run_lorenz_episode(rho))
        self.rng = rng or np.random.default_rng()
        v0 = np.array([28.0, 0.8], dtype=float)
        v0 = self._project(v0)
        self.state = MetaState(k=0, v=v0)

    # --- SPSA helpers ---

    def _sample_delta(self) -> np.ndarray:
        return self.rng.choice([-1.0, 1.0], size=2)

    def _project(self, v: np.ndarray) -> np.ndarray:
        (r_min, r_max), (a_min, a_max) = self.spsa_cfg.bounds
        out = np.array(v, dtype=float)
        out[0] = float(np.clip(out[0], r_min, r_max))
        out[1] = float(np.clip(out[1], a_min, a_max))
        return out

    # --- Evaluation pipeline ---

    def evaluate_J(self, v: np.ndarray) -> Dict[str, float]:
        rho, alpha = float(v[0]), float(v[1])
        ep = self.run_episode(rho)
        section_hits = np.asarray(ep.get("section_hits", np.zeros((0, 2))), dtype=float)
        energy = float(ep.get("raw_metrics", {}).get("energy", 0.0))

        if section_hits.shape[0] < 10:
            return {"J": 1e9, "lam2_abs": 1.0, "gamma": 0.0, "D2": 0.0, "energy": energy}

        P, _ = build_ulam(section_hits, nx=25, ny=25, bounds=None)
        gated = left_right_gate(P, nx=25, ny=25, alpha=alpha)
        op_stats = analyze_operator(gated, mean_return=1.0, nx=25, ny=25)
        lam2_abs = float(op_stats.get("abs_lambda2", 0.0))
        gamma = float(op_stats.get("gamma", 0.0))

        symbols = build_symbolic_sequence(section_hits, coding_spec="x_sign")
        frac = build_fractalhedron_k(symbols, k=2, Q=[0.0, 2.0])
        D2 = float(frac["D_q"].get(2.0, 0.0))

        J = compute_cost(lam2_abs, gamma, D2, energy)
        return {"J": J, "lam2_abs": lam2_abs, "gamma": gamma, "D2": D2, "energy": energy}

    # --- Meta step ---

    def step(self) -> Dict[str, float]:
        cfg = self.spsa_cfg
        k = self.state.k
        a_k, c_k = cfg.gains(k)
        delta = self._sample_delta()

        v = self.state.v
        v_plus = self._project(v + c_k * delta)
        v_minus = self._project(v - c_k * delta)

        res_plus = self.evaluate_J(v_plus)
        res_minus = self.evaluate_J(v_minus)

        J_plus = res_plus["J"]
        J_minus = res_minus["J"]
        g_hat = ((J_plus - J_minus) / (2 * c_k)) * delta
        v_new_raw = v - a_k * (cfg.precond * g_hat)
        v_new = self._project(v_new_raw)

        J_mid = 0.5 * (J_plus + J_minus)
        self.state.history.append(
            {
                "k": float(k),
                "rho": float(v[0]),
                "alpha": float(v[1]),
                "J": float(J_mid),
                "g_rho": float(g_hat[0]),
                "g_alpha": float(g_hat[1]),
            }
        )
        self.state.v = v_new
        self.state.k = k + 1

        return {
            "k": k,
            "v_old": v,
            "v_new": v_new,
            "J_plus": J_plus,
            "J_minus": J_minus,
            "g_hat": g_hat,
            "res_plus": res_plus,
            "res_minus": res_minus,
        }

    def state_summary(self) -> Dict[str, object]:
        return {"k": self.state.k, "v": self.state.v.tolist(), "history": self.state.history}

