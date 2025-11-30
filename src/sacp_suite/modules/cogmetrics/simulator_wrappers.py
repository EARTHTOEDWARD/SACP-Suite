"""Helpers that wrap plugin dynamics with parameter forcing."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from sacp_suite.core.plugin_api import registry
from sacp_suite.modules.cogmetrics.experiments import SimulatorFn


def make_param_forced_simulator(
    *,
    model_key: str,
    param_name: str,
    base_value: float,
    amp: float,
    dt: float,
    extra_params: Dict[str, Any] | None = None,
) -> SimulatorFn:
    """Return a simulator callable that forces a model parameter per timestep.

    Each timestep adjusts ``param_name`` to ``base_value + amp * u[t]`` before
    integrating a single RK4 step. The returned outputs follow the simulator
    convention: shape ``(T, d)`` with ``d`` equal to the model state dimension.
    """

    extras = dict(extra_params or {})

    def simulator(inputs: np.ndarray) -> np.ndarray:
        u = np.asarray(inputs, dtype=float)
        if u.ndim != 1:
            raise ValueError("inputs must be a 1D array of forcings")

        params = {**extras, param_name: base_value}
        dyn = registry.create_dynamics(model_key, **params)
        state = dyn.default_state().astype(float)
        output = np.empty((u.shape[0], state.shape[0]), dtype=float)

        for idx, drive in enumerate(u):
            dyn.params[param_name] = base_value + amp * float(drive)
            state = dyn.rk4_step(state, dt)
            output[idx] = state
        return output

    return simulator
