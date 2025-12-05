"""
ADR–Hemostasis (Grytsay surrogate) on a 4-site Instrumented ADR ring.

This wraps InstrumentedADR and exposes a plugin_api.BaseDynamics interface so
the SACP/ABTC stack can treat it like Lorenz/Rössler. Plasticity on the
diffusive couplings k_i is preserved via InstrumentedADR.

Site semantics (x indexes):
    0: prostanoid_balance   (P vs Tx tone)
    1: thrombus_shock       (acute thrombosis / embolic mode)
    2: ldl_plaque           (LDL / plaque burden)
    3: cytokine_field       (inflammatory milieu)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np

from sacp_suite.core.plugin_api import BaseDynamics, registry
from sacp_suite.modules.chemistry.instrumented_adr import InstrumentedADR


MODEL_ID = "adr_hemostasis"
DISPLAY_NAME = "ADR–Hemostasis (Grytsay surrogate)"


@dataclass
class ADRHemostasisParams:
    # Core integration
    dt: float = 0.01

    # Baseline Duffing curvature and damping
    a_base: float = 1.0
    gamma_base: float = 0.2

    # Per-site heterogeneity multipliers (a_i = a_base * a_scale[i], etc.)
    a_scale: Sequence[float] = (1.0, 1.1, 0.9, 1.05)
    gamma_scale: Sequence[float] = (1.0, 1.1, 1.0, 0.95)

    # Resource loop
    mu: float = 0.08
    sigma: float = 0.6
    eta: float = 0.5

    # External drive (bifurcation knob)
    F: float = 0.25
    omega: float = 1.0

    # Coupling / plasticity (self-tuning)
    k_init: float = 0.4
    k_min: float = 0.0
    k_max: float = 1.5
    plasticity_enabled: bool = True
    eta_plast: float = 1e-3
    R_target: float = 0.6
    plasticity_every: int = 10
    neighborhood_radius: int = 1

    # RNG
    seed: Optional[int] = None


def _make_params_dict(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    defaults = asdict(ADRHemostasisParams())
    if overrides:
        defaults.update(overrides)
    return defaults


class ADRHemostasisDynamics(BaseDynamics):
    """
    12D state: [x0..x3, v0..v3, r0..r3] driven by InstrumentedADR with plastic k_i.
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.params = _make_params_dict(self.params)
        self.N = 4
        self._build_adr()

    # ------------------------------------------------------------------
    # BaseDynamics interface
    # ------------------------------------------------------------------

    def get_default_params(self) -> dict:
        return _make_params_dict(None)

    def default_state(self) -> np.ndarray:
        return self._flat_state()

    def derivative(self, s: np.ndarray, t: float = 0.0) -> np.ndarray:  # noqa: ARG002
        """
        Continuous-time RHS (no plasticity updates here). Used mainly for
        compatibility; the simulate override is the preferred path.
        """
        N = self.N
        x = s[0:N]
        v = s[N:2 * N]
        r = s[2 * N:3 * N]

        lap = np.roll(x, -1) + np.roll(x, 1) - 2.0 * x
        r_prev = np.roll(r, 1)
        k = self.adr.k  # current coupling snapshot

        dx = v
        dv = (
            -self.gamma * v
            + self.a * x
            - x**3
            + k * lap
            + self.eta * r_prev
            + self.F * np.cos(self.omega * float(self.adr.t))
        )
        dr = -self.mu * r + self.sigma * x**2
        return np.concatenate([dx, dv, dr])

    @property
    def name(self) -> str:
        return DISPLAY_NAME

    @property
    def state_labels(self) -> list[str]:
        return [
            "x_prostanoid",
            "x_thrombus",
            "x_ldl_plaque",
            "x_cytokine",
            "v_prostanoid",
            "v_thrombus",
            "v_ldl_plaque",
            "v_cytokine",
            "r_prostanoid",
            "r_thrombus",
            "r_ldl_plaque",
            "r_cytokine",
        ]

    def simulate(self, T: float = 50.0, dt: float = 0.01, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Override to leverage InstrumentedADR (plasticity-aware).
        """
        n = int(max(T / dt, 1))

        # Optionally reset state to provided x0
        if x0 is not None:
            self._set_flat_state(x0)

        # Temporarily adjust dt if needed
        original_dt = self.adr.dt
        if dt != original_dt:
            self.adr.dt = float(dt)

        log = self.adr.run(n_steps=n, log_every=1)
        traj = np.concatenate([log["x"], log["v"], log["r"]], axis=1)

        # restore dt
        self.adr.dt = original_dt
        return traj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_adr(self) -> None:
        p = ADRHemostasisParams(**self.params)

        a_scale = np.asarray(p.a_scale, dtype=float)
        gamma_scale = np.asarray(p.gamma_scale, dtype=float)
        if a_scale.shape[0] != self.N or gamma_scale.shape[0] != self.N:
            raise ValueError(f"a_scale and gamma_scale must have length {self.N}.")

        self.a = p.a_base * a_scale
        self.gamma = p.gamma_base * gamma_scale
        self.mu = float(p.mu)
        self.sigma = float(p.sigma)
        self.eta = float(p.eta)
        self.F = float(p.F)
        self.omega = float(p.omega)

        self.adr = InstrumentedADR(
            N=self.N,
            dt=p.dt,
            a=self.a,
            gamma=self.gamma,
            mu=self.mu,
            sigma=self.sigma,
            eta=self.eta,
            F=self.F,
            omega=self.omega,
            k_init=p.k_init,
            k_min=p.k_min,
            k_max=p.k_max,
            plasticity_enabled=p.plasticity_enabled,
            eta_plast=p.eta_plast,
            R_target=p.R_target,
            plasticity_every=p.plasticity_every,
            neighborhood_radius=p.neighborhood_radius,
            seed=p.seed,
        )

    def _flat_state(self) -> np.ndarray:
        return np.concatenate([self.adr.x, self.adr.v, self.adr.r]).astype(float, copy=True)

    def _set_flat_state(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        expected = 3 * self.N
        if y.shape[0] != expected:
            raise ValueError(f"Expected state length {expected}, got {y.shape[0]}")
        self.adr.x = y[0:self.N].copy()
        self.adr.v = y[self.N:2 * self.N].copy()
        self.adr.r = y[2 * self.N:3 * self.N].copy()


registry.register_dynamics(MODEL_ID, ADRHemostasisDynamics)
