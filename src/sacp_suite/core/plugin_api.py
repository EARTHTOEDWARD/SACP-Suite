"""Plugin primitives for SACP Suite dynamics and services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Optional

import numpy as np


class BaseDynamics(ABC):
    """Abstract base for a dynamical system plugin."""

    params: Dict[str, float]

    def __init__(self, **params: float) -> None:
        defaults = self.get_default_params().copy()
        defaults.update(params or {})
        self.params = defaults

    @abstractmethod
    def get_default_params(self) -> Dict[str, float]:
        """Return the default parameter dictionary."""

    @abstractmethod
    def default_state(self) -> np.ndarray:
        """Return a default state vector."""

    @abstractmethod
    def derivative(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Return the instantaneous derivative at state `state`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human friendly name."""

    @property
    @abstractmethod
    def state_labels(self) -> List[str]:
        """Return labels for each state dimension."""

    def rk4_step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Single Runge-Kutta step."""

        f = self.derivative
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, T: float = 50.0, dt: float = 0.01, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Integrate forward for duration `T` with timestep `dt`."""

        n = int(max(T / dt, 1))
        x = self.default_state() if x0 is None else x0.copy()
        traj = np.zeros((n, x.shape[0]), dtype=float)
        for i in range(n):
            traj[i] = x
            x = self.rk4_step(x, dt)
        return traj


class PluginRegistry:
    """Simple in-memory registry for dynamics plugins."""

    def __init__(self) -> None:
        self._dyn: Dict[str, Type[BaseDynamics]] = {}

    def register_dynamics(self, key: str, cls: Type[BaseDynamics]) -> None:
        self._dyn[key] = cls

    def list_dynamics(self) -> List[str]:
        return sorted(self._dyn.keys())

    def create_dynamics(self, key: str, **params: Any) -> BaseDynamics:
        if key not in self._dyn:
            raise KeyError(f"Unknown dynamics '{key}'")
        return self._dyn[key](**params)


registry = PluginRegistry()
