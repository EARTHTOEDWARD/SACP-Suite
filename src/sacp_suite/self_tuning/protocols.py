"""Protocols used by the SACP self-tuning adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sacp_suite.selftuner.types import TuningAction


class SelfTunableSystem(Protocol):
    """Minimal interface a SACP system must satisfy for self-tuning."""

    @property
    def id(self) -> str:
        ...

    @property
    def dt(self) -> float:
        ...

    def get_state_vector(self) -> np.ndarray:
        ...

    def apply_tuning_action(self, action: TuningAction) -> None:
        ...
