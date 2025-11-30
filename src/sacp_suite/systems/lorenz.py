"""Self-tunable Lorenz attractor demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sacp_suite.self_tuning.protocols import SelfTunableSystem
from sacp_suite.selftuner.types import TuningAction


@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


class LorenzSystem(SelfTunableSystem):
    """Lorenz 63 attractor instrumented for self-tuning experiments."""

    def __init__(
        self,
        id_: str = "lorenz",
        dt: float = 0.01,
        params: LorenzParams | None = None,
    ) -> None:
        self._id = id_
        self._dt = float(dt)
        self.params = params or LorenzParams()
        self.state = np.array([1.0, 1.0, 1.0], dtype=float)
        self.spectral_radius = 1.0  # scales rho
        self.gain = 1.0  # scales entire vector field

    # ------------------------------------------------------------------ #
    # SelfTunableSystem protocol
    # ------------------------------------------------------------------ #

    @property
    def id(self) -> str:
        return self._id

    @property
    def dt(self) -> float:
        return self._dt

    def get_state_vector(self) -> np.ndarray:
        return self.state

    def apply_tuning_action(self, action: TuningAction) -> None:
        if action.spectral_radius_delta:
            self.spectral_radius += action.spectral_radius_delta
        if action.gain_delta:
            self.gain += action.gain_delta
        if action.target_spectral_radius is not None:
            self.spectral_radius = float(action.target_spectral_radius)
        if action.target_gain is not None:
            self.gain = float(action.target_gain)
        self.spectral_radius = float(np.clip(self.spectral_radius, 0.1, 10.0))
        self.gain = float(np.clip(self.gain, 0.1, 10.0))

    # ------------------------------------------------------------------ #
    # Dynamics helpers
    # ------------------------------------------------------------------ #

    def _rhs(self, x: np.ndarray) -> np.ndarray:
        sigma = self.params.sigma
        rho = self.params.rho * self.spectral_radius
        beta = self.params.beta
        x_val, y_val, z_val = x
        dx = sigma * (y_val - x_val)
        dy = x_val * (rho - z_val) - y_val
        dz = x_val * y_val - beta * z_val
        return self.gain * np.array([dx, dy, dz], dtype=float)

    def _rk4_step(self, x: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._rhs(x)
        k2 = self._rhs(x + 0.5 * dt * k1)
        k3 = self._rhs(x + 0.5 * dt * k2)
        k4 = self._rhs(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self) -> None:
        self.state = self._rk4_step(self.state, self._dt)

    def step_pure(self, state: np.ndarray, dt: float) -> np.ndarray:
        return self._rk4_step(np.asarray(state, dtype=float), dt)
