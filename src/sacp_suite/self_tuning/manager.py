"""High-level manager bridging SACP systems to the self-tuner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import numpy as np

from sacp_suite.self_tuning.protocols import SelfTunableSystem
from sacp_suite.selftuner.types import (
    LyapunovEstimate,
    TuningContext,
    TuningAction,
    ChaosBand,
)
from sacp_suite.selftuner.lyapunov import (
    QRStreamingLyapunovEstimator,
    TangentMapFn,
)
from sacp_suite.selftuner.fd_tangent import make_fd_tangent_map_from_step
from sacp_suite.selftuner.controller import SimpleChaosBandController
from sacp_suite.selftuner.filters import IdentityPerceptionFilters, PerceptionFilters

SystemStepFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class SelfTuningState:
    last_estimate: Optional[LyapunovEstimate] = None
    last_action: Optional[TuningAction] = None


class SelfTuningManager:
    """Bridge between a SACP system and the vendored self-tuner."""

    def __init__(
        self,
        system: SelfTunableSystem,
        state_dim: int,
        system_step_fn: Optional[SystemStepFn] = None,
        tangent_map: Optional[TangentMapFn] = None,
        chaos_band: ChaosBand = ChaosBand(lower=0.0, upper=0.5),
        n_exponents: int = 1,
        k_spectral_radius: float = 0.1,
        k_gain: float = 0.0,
        window: int = 2000,
        perception_filters: Optional[PerceptionFilters] = None,
        modality: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.system = system
        self.state_dim = state_dim
        self.chaos_band = chaos_band
        self.modality = modality
        self._extras = extras or {}

        if tangent_map is None and system_step_fn is not None:
            tangent_map = make_fd_tangent_map_from_step(system_step_fn)

        self.estimator = QRStreamingLyapunovEstimator(
            state_dim=state_dim,
            n_exponents=n_exponents,
            tangent_map=tangent_map,
            window=window,
        )

        self.controller = SimpleChaosBandController(
            target_band=chaos_band,
            k_spectral_radius=k_spectral_radius,
            k_gain=k_gain,
        )

        self.filters = perception_filters or IdentityPerceptionFilters()

        self.context = TuningContext(
            system_id=system.id,
            modality=modality,
            extras=self._extras,
        )

        self.state = SelfTuningState()

    def reset(self) -> None:
        self.estimator.reset()
        self.state = SelfTuningState()

    def update_for_current_state(self) -> None:
        x_t = self.system.get_state_vector()
        dt = self.system.dt
        estimate = self.estimator.update(x_t, dt)
        action = self.controller.tune(estimate, self.context)
        self.system.apply_tuning_action(action)
        self.filters.apply(x_t, action, self.context)
        self.state.last_estimate = estimate
        self.state.last_action = action

    def get_latest_lambda_max(self) -> Optional[float]:
        if not self.state.last_estimate or self.state.last_estimate.values.size == 0:
            return None
        return float(self.state.last_estimate.values[0])

    def get_latest_regime(self) -> Optional[str]:
        if not self.state.last_action or not self.state.last_action.metadata:
            return None
        return self.state.last_action.metadata.get("regime")
