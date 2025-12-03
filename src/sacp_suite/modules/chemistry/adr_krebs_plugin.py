"""ABCP/SACP plugin for the ADR–Krebs tile."""

from __future__ import annotations

import numpy as np

from sacp_suite.core.plugin_api import BaseDynamics, registry


class ADRKrebsDynamics(BaseDynamics):
    """12D state: [x1..x4, v1..v4, r1..r4]"""

    def get_default_params(self) -> dict:
        return {
            "alpha": 1.0,
            "a_TCA": 1.10,
            "a_NADH": 1.00,
            "a_dPsi": 1.20,
            "a_ATP": 0.90,
            "gamma_TCA": 0.22,
            "gamma_NADH": 0.20,
            "gamma_dPsi": 0.18,
            "gamma_ATP": 0.16,
            "mu": 0.15,
            "sigma": 0.60,
            "eta_base": 0.50,
            "kappa": 0.40,
        }

    def default_state(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        x = rng.uniform(-1.0, 1.0, size=4)
        v = rng.uniform(-0.1, 0.1, size=4)
        r = np.zeros(4, dtype=float)
        return np.concatenate([x, v, r])

    def derivative(self, s: np.ndarray, t: float = 0.0) -> np.ndarray:  # noqa: ARG002
        p = self.params
        a = np.array([p["a_TCA"], p["a_NADH"], p["a_dPsi"], p["a_ATP"]], dtype=float)
        gamma = np.array([p["gamma_TCA"], p["gamma_NADH"], p["gamma_dPsi"], p["gamma_ATP"]], dtype=float)
        mu = float(p["mu"])
        sigma = float(p["sigma"])
        eta = float(p["eta_base"]) * float(p["alpha"])
        kappa = float(p["kappa"])

        x = s[0:4]
        v = s[4:8]
        r = s[8:12]

        lap = np.roll(x, -1) + np.roll(x, 1) - 2.0 * x
        r_prev = np.roll(r, 1)

        dx = v
        dv = -gamma * v + a * x - x**3 + kappa * lap + eta * r_prev
        dr = -mu * r + sigma * x**2

        return np.concatenate([dx, dv, dr])

    @property
    def name(self) -> str:
        return "ADR–Krebs v1"

    @property
    def state_labels(self) -> list[str]:
        return [
            "x_TCA",
            "x_NADH",
            "x_dPsi",
            "x_ATP",
            "v_TCA",
            "v_NADH",
            "v_dPsi",
            "v_ATP",
            "r_TCA",
            "r_NADH",
            "r_dPsi",
            "r_ATP",
        ]


registry.register_dynamics("adr_krebs", ADRKrebsDynamics)
