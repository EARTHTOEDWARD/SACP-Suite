from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np
import numpy.typing as npt


@dataclass
class ParamSpec:
    """Parameter metadata for UI and API exposure."""

    name: str
    label: str
    default: float
    min: float
    max: float
    log_scale: bool = False
    step: float | None = None


@dataclass
class DynamicsMetadata:
    id: str
    label: str
    state_dim: int
    param_specs: Dict[str, ParamSpec]
    kind: Literal["ode", "fde"]
    has_offset: bool
    supports_bifurcation_scan: bool
    supports_basin_scan: bool
    supports_complexity: bool


class BaseDynamics(ABC):
    """
    Abstract base for both integer-order ODEs and fractional dynamics.

    This mirrors the sketch in the user request: implementors supply
    defaults, metadata, RHS, and optional decompositions.
    """

    id: str
    label: str
    state_dim: int
    kind: Literal["ode", "fde"] = "ode"

    @abstractmethod
    def default_params(self) -> Dict[str, float]:
        ...

    @abstractmethod
    def param_specs(self) -> Dict[str, ParamSpec]:
        ...

    @abstractmethod
    def default_state(self) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def rhs(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        params: Dict[str, float],
    ) -> npt.NDArray[np.float64]:
        """
        Returns f(x, t; params). For fractional systems this is the RHS used by
        the fractional integrator D_c^q x = f(x, t).
        """
        ...

    def rhs_decomposed(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        params: Dict[str, float],
    ) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Optional chemical decomposition of the RHS.
        Default: a single 'total' component.
        """
        return {"total": self.rhs(t, x, params)}

    def orders(self, params: Dict[str, float]) -> npt.NDArray[np.float64]:
        """Fractional orders per state; default is classic ODE (all ones)."""
        return np.ones(self.state_dim)

    def metadata(self) -> DynamicsMetadata:
        specs = self.param_specs()
        return DynamicsMetadata(
            id=self.id,
            label=self.label,
            state_dim=self.state_dim,
            param_specs=specs,
            kind=self.kind,
            has_offset="m" in specs,
            supports_bifurcation_scan=True,
            supports_basin_scan=True,
            supports_complexity=True,
        )

    # Convenience hook to keep compatibility with the legacy registry API.
    @property
    def name(self) -> str:
        return self.label

    @property
    def state_labels(self) -> list[str]:
        return [f"x{i}" for i in range(self.state_dim)]

    def derivative(self, state: npt.NDArray[np.float64], t: float = 0.0) -> npt.NDArray[np.float64]:
        return self.rhs(t, state, self.default_params())
