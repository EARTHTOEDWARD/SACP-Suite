"""
Krebs + ADR–Krebs models for SACP / ABCP
========================================

This file defines two small dynamical systems:

1. KrebsV1:
   A coarse 4‑variable ODE capturing the TCA / Krebs energy loop:
       P   ~ TCA throughput / flux
       N   ~ NADH / NAD⁺ redox ratio
       Psi ~ mitochondrial membrane potential ΔΨ
       A   ~ ATP / ADP ratio

   The main control parameter `alpha4` plays the role of ATP
   dissipation / load (loosely echoing Grytsay's α₄).

2. ADRKrebsTile:
   A 4‑site Autocatalytic Duffing Ring (ADR) whose sites are mapped to
   the same four coarse variables, with a resource loop r_i around the
   ring. This is meant as a canonical ADR tile for metabolic chemistry.

Both classes are deliberately *self-contained*: they don't depend on
SACP / ABTC internals. You can wrap them in a BaseDynamics plugin or
call them directly from notebooks and operator code.

Dependencies: numpy
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _sat_exc(x: float | np.ndarray, K: float) -> np.ndarray:
    """Simple excitatory saturation x / (K + x), safe for small x."""
    x_arr = np.asarray(x, dtype=float)
    return x_arr / (K + x_arr + 1e-12)


def _sat_inh(x: float | np.ndarray, K: float) -> np.ndarray:
    """Simple inhibitory saturation K / (K + x)."""
    x_arr = np.asarray(x, dtype=float)
    return K / (K + x_arr + 1e-12)


# ---------------------------------------------------------------------
# 1. Coarse 4D Krebs model
# ---------------------------------------------------------------------


@dataclass
class KrebsV1Params:
    # TCA flux / substrate supply
    J_in: float = 1.0       # effective substrate influx
    K_N: float = 0.5        # NADH inhibition half-saturation
    K_A: float = 1.0        # ATP inhibition half-saturation

    # Kinetics
    k_P_decay: float = 0.2  # TCA flux relaxation
    k_P_to_N: float = 0.6   # NADH production from TCA flux
    k_N_decay: float = 0.2  # NADH oxidation baseline

    # Electron transport & membrane
    k_et: float = 1.0       # NADH -> ET chain rate prefactor
    psi_max: float = 1.0    # saturation for ΔΨ
    k_pump: float = 0.8     # ET -> ΔΨ pumping
    k_leak: float = 0.3     # ΔΨ leak
    k_atp_psi: float = 0.7  # ΔΨ consumption by ATP synthase

    # ATP synthesis & use
    A_max: float = 2.0      # saturation for ATP/ADP ratio
    k_syn: float = 0.8      # ATP synthesis from ΔΨ
    k_use_base: float = 0.3 # baseline ATP use
    alpha4: float = 0.0     # "extra" ATP dissipation / load

    # Numerical
    dt: float = 0.01


class KrebsV1:
    """
    Minimal 4‑variable Krebs/TCA model.

    State vector y = [P, N, Psi, A], all dimensionless and O(1).

        P   : coarse TCA throughput
        N   : NADH/NAD⁺ redox level
        Psi : mitochondrial membrane potential ΔΨ
        A   : ATP/ADP ratio

    Control knob:
        alpha4 (inside params) -> increases ATP usage rate k_use.

    This is NOT a faithful copy of Grytsay's 19‑D system; it's a
    coarse attractor-compatible surrogate intended to share the same
    qualitative bifurcation story (period‑doubling via ATP load).
    """

    state_dim: int = 4

    def __init__(self, params: Optional[KrebsV1Params] = None):
        self.p = params or KrebsV1Params()

    # -------- core dynamics --------

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        P, N, Psi, A = y
        p = self.p

        # Guard some variables to avoid pathological divisions
        N_pos = max(N, 0.0)
        Psi_clip = np.clip(Psi, 0.0, p.psi_max)
        A_clip = np.clip(A, 0.0, p.A_max)

        # TCA flux, inhibited by high NADH and high ATP
        J_tca = (
            p.J_in
            * _sat_inh(N_pos, p.K_N)
            * _sat_inh(A_clip, p.K_A)
        )

        # Electron transport: driven by NADH, opposed by high ΔΨ
        J_et = p.k_et * N_pos * (1.0 - Psi_clip / p.psi_max)

        # ATP synthase flux: on when ΔΨ is high and ATP is not yet saturated
        V_atp = Psi_clip * (1.0 - A_clip / p.A_max)

        # Effective ATP usage
        k_use = p.k_use_base + p.alpha4

        dP = J_tca - p.k_P_decay * P
        dN = p.k_P_to_N * J_tca - J_et - p.k_N_decay * N
        dPsi = p.k_pump * J_et - p.k_leak * Psi - p.k_atp_psi * V_atp
        dA = p.k_syn * V_atp - k_use * A

        return np.array([dP, dN, dPsi, dA], dtype=float)

    # -------- integrator helpers --------

    def step_euler(self, y: np.ndarray, t: float) -> Tuple[np.ndarray, float]:
        dt = self.p.dt
        dy = self.rhs(t, y)
        return y + dt * dy, t + dt

    def step_rk4(self, y: np.ndarray, t: float) -> Tuple[np.ndarray, float]:
        dt = self.p.dt
        f = self.rhs

        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)

        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_new, t + dt

    def simulate(
        self,
        y0: np.ndarray,
        n_steps: int,
        method: str = "rk4",
        discard: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Run the ODE for n_steps; return dict with t and trajectory.

        discard: how many initial steps to drop from the returned arrays
                 (e.g. to remove transient before attractor analysis).
        """
        y = np.asarray(y0, dtype=float).copy()
        if y.shape != (4,):
            raise ValueError("y0 must be a length‑4 vector [P, N, Psi, A].")

        t = 0.0
        if method == "rk4":
            step = self.step_rk4
        elif method == "euler":
            step = self.step_euler
        else:
            raise ValueError(f"Unknown method '{method}'")

        traj = np.zeros((n_steps, 4), dtype=float)
        times = np.zeros(n_steps, dtype=float)

        for i in range(n_steps):
            traj[i] = y
            times[i] = t
            y, t = step(y, t)

        if discard > 0:
            traj = traj[discard:]
            times = times[discard:]

        return {"t": times, "y": traj, "params": asdict(self.p)}

    # -------- SACP-style conveniences --------

    @staticmethod
    def state_dict(y: np.ndarray) -> Dict[str, float]:
        """Map state vector -> named chemical observables."""
        P, N, Psi, A = y
        return {
            "P_TCA": float(P),
            "NADH_ratio": float(N),
            "DeltaPsi": float(Psi),
            "ATP_ADP": float(A),
        }


# ---------------------------------------------------------------------
# 2. ADR–Krebs tile (4‑site ADR ring with metabolic labels)
# ---------------------------------------------------------------------


@dataclass
class ADRKrebsParams:
    # Duffing potentials per site (a_i x - x^3)
    a: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.9, 1.1, 0.95], dtype=float)
    )
    # Damping per site
    gamma: np.ndarray = field(
        default_factory=lambda: np.array([0.18, 0.20, 0.22, 0.24], dtype=float)
    )

    # Resource loop
    mu_base: float = 0.08      # resource decay baseline
    sigma: float = 0.5         # resource production from x_i^2
    alpha4: float = 0.0        # extra ATP dissipation → effective mu

    # Coupling and autocatalysis
    k_init: float = 0.35       # diffusive coupling strength κ
    eta: float = 0.6           # catalytic drive around ring

    # Global drive (e.g. respiratory substrate supply)
    F: float = 0.18
    omega: float = 0.8

    # Numerical
    dt: float = 0.01

    # Random initialisation
    seed: Optional[int] = 1


class ADRKrebsTile:
    """
    4‑site Autocatalytic Duffing Ring for the Krebs cycle.

    Sites / coordinates:
        x[0] ↔ TCA throughput          (P)
        x[1] ↔ NADH / NAD⁺ redox level (N)
        x[2] ↔ membrane potential ΔΨ   (Psi)
        x[3] ↔ ATP / ADP ratio         (A)

    Resource variables r[i] circulate clockwise: r[i] is produced by
    site i and drives site (i+1 mod 4). The single control knob
    alpha4 (in params) increases the global resource decay:

        mu_eff = mu_base + alpha4

    which plays the role of an ATP dissipation / load parameter
    (loosely analogous to α₄ in Grytsay).

    Equations (Euler‑integrated):

        v_i = dx_i/dt

        dv_i/dt = -gamma_i * v_i
                  + a_i * x_i - x_i^3
                  + k * Laplacian(x)_i
                  + eta * r_{i-1}
                  + F * cos(omega * t)

        dr_i/dt = -mu_eff * r_i + sigma * x_i^2

    with ring Laplacian:
        Laplacian(x)_i = x_{i+1} + x_{i-1} - 2 x_i
    """

    N: int = 4

    def __init__(self, params: Optional[ADRKrebsParams] = None):
        self.p = params or ADRKrebsParams()

        # Ensure arrays
        self.a = np.asarray(self.p.a, dtype=float)
        self.gamma = np.asarray(self.p.gamma, dtype=float)
        if self.a.shape != (self.N,) or self.gamma.shape != (self.N,):
            raise ValueError("a and gamma must be length‑4 arrays.")

        self.dt = float(self.p.dt)
        self.mu_eff = float(self.p.mu_base + self.p.alpha4)

        rng = np.random.default_rng(self.p.seed)
        self.x = rng.uniform(-1.0, 1.0, size=self.N)
        self.v = rng.uniform(-0.1, 0.1, size=self.N)
        self.r = np.zeros(self.N, dtype=float)

        self.k = float(self.p.k_init)
        self.eta = float(self.p.eta)
        self.F = float(self.p.F)
        self.omega = float(self.p.omega)

        self.t = 0.0

    # -------- core updates --------

    def _laplacian(self, x: np.ndarray) -> np.ndarray:
        return np.roll(x, -1) + np.roll(x, 1) - 2.0 * x

    def step(self) -> None:
        """One Euler step of the ADR–Krebs dynamics."""
        dt = self.dt
        x, v, r = self.x, self.v, self.r

        lap = self._laplacian(x)
        r_prev = np.roll(r, 1)  # r_{i-1} due to roll(+1)

        accel = (
            -self.gamma * v
            + self.a * x
            - x**3
            + self.k * lap
            + self.eta * r_prev
            + self.F * np.cos(self.omega * self.t)
        )

        v_new = v + dt * accel
        x_new = x + dt * v_new
        r_new = r + dt * (-self.mu_eff * r + self.p.sigma * x**2)

        self.v = v_new
        self.x = x_new
        self.r = r_new

        self.t += dt

    # -------- simulation / logging --------

    def simulate(
        self,
        n_steps: int,
        log_every: int = 1,
        discard: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Run ADR–Krebs ring; return trajectory logs.

        Returns a dict with:
            t       : (T,) time samples
            x       : (T, 4) positions
            v       : (T, 4) velocities
            r       : (T, 4) resources
        """
        log_every = max(1, int(log_every))
        xs, vs, rs, ts = [], [], [], []

        for step_idx in range(n_steps):
            if step_idx % log_every == 0:
                xs.append(self.x.copy())
                vs.append(self.v.copy())
                rs.append(self.r.copy())
                ts.append(self.t)
            self.step()

        X = np.array(xs)
        V = np.array(vs)
        R = np.array(rs)
        T = np.array(ts)

        if discard > 0:
            X = X[discard:]
            V = V[discard:]
            R = R[discard:]
            T = T[discard:]

        return {
            "t": T,
            "x": X,
            "v": V,
            "r": R,
            "params": asdict(self.p),
        }

    # -------- SACP-style conveniences --------

    @staticmethod
    def x_to_dict(x: np.ndarray) -> Dict[str, float]:
        """
        Map the 4‑vector x into named metabolic observables.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (4,):
            raise ValueError("x must be length‑4.")
        return {
            "P_TCA": float(x[0]),
            "NADH_ratio": float(x[1]),
            "DeltaPsi": float(x[2]),
            "ATP_ADP": float(x[3]),
        }
