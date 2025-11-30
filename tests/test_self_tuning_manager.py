import numpy as np

from sacp_suite.self_tuning.manager import SelfTuningManager
from sacp_suite.self_tuning.protocols import SelfTunableSystem
from sacp_suite.selftuner.types import ChaosBand, TuningAction


class DummySystem(SelfTunableSystem):
    def __init__(self, dt: float = 0.01):
        self._dt = dt
        self.state = np.ones(2, dtype=float)
        self.spectral_radius = 1.0

    @property
    def id(self) -> str:
        return "dummy"

    @property
    def dt(self) -> float:
        return self._dt

    def get_state_vector(self) -> np.ndarray:
        return self.state

    def apply_tuning_action(self, action: TuningAction) -> None:
        self.spectral_radius += action.spectral_radius_delta

    def step(self) -> None:
        self.state = self.state + 0.01

    def step_pure(self, state: np.ndarray, dt: float) -> np.ndarray:
        return state + 0.01


def test_manager_updates_state():
    system = DummySystem()
    mgr = SelfTuningManager(
        system=system,
        state_dim=2,
        system_step_fn=system.step_pure,
        chaos_band=ChaosBand(lower=-1e-3, upper=1e-3),
        window=2,
    )
    system.step()
    mgr.update_for_current_state()
    assert mgr.state.last_action is not None
