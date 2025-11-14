"""Lorenz-63 dynamics plugin."""

from __future__ import annotations

import numpy as np

from sacp_suite.core.plugin_api import BaseDynamics, registry


class Lorenz63(BaseDynamics):
    def get_default_params(self) -> dict:
        return {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}

    def default_state(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def derivative(self, s: np.ndarray, t: float = 0.0) -> np.ndarray:  # noqa: ARG002
        x, y, z = s
        p = self.params
        dx = p["sigma"] * (y - x)
        dy = x * (p["rho"] - z) - y
        dz = x * y - p["beta"] * z
        return np.array([dx, dy, dz], dtype=float)

    @property
    def name(self) -> str:
        return "Lorenz 1963"

    @property
    def state_labels(self) -> list[str]:
        return ["X", "Y", "Z"]


registry.register_dynamics("lorenz63", Lorenz63)
