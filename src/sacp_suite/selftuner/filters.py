"""Optional perception filters for the self-tuner."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from sacp_suite.selftuner.types import TuningAction, TuningContext


class PerceptionFilters(Protocol):
    """Optional hooks invoked after a tuning action is applied."""

    def apply(self, state: np.ndarray, action: TuningAction, context: TuningContext) -> None:  # pragma: no cover - interface
        ...


class IdentityPerceptionFilters:
    """Default filters that perform no post-processing."""

    def apply(self, state: np.ndarray, action: TuningAction, context: TuningContext) -> None:
        return None
