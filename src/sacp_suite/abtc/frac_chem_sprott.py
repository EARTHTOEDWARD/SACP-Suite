from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

from sacp_suite.abtc.base import BaseDynamics, ParamSpec


@dataclass
class FracChemSprottDynamics(BaseDynamics):
    """
    Fractional-order chemical Sprott system with offset boosting.

    D_c^q x = k1 + k4*y*z_eff - k6*y
    D_c^q y = k2 + k5*x^2 - k7*x - k9*y
    D_c^q z = k3 - k8*x
    """

    id: str = "frac_chem_sprott"
    label: str = "Fractional chemical Sprott (hidden attractor)"
    state_dim: int = 3
    kind: str = "fde"

    def default_params(self) -> Dict[str, float]:
        return {
            "q": 0.95,
            "k1": 0.006,
            "k2": 9.0,
            "k3": 13.0,
            "k4": 1.0,
            "k5": 1.0,
            "k6": 3.0,
            "k7": 6.0,
            "k8": 4.0,
            "k9": 1.0,
            "m": 0.0,
        }

    def param_specs(self) -> Dict[str, ParamSpec]:
        p = self.default_params()
        return {
            "q": ParamSpec("q", "fractional order q", p["q"], 0.55, 1.0, step=0.01),
            "k1": ParamSpec("k1", "k1 (source X)", p["k1"], 0.0, 0.07),
            "k2": ParamSpec("k2", "k2 (source Y)", p["k2"], 0.0, 20.0),
            "k3": ParamSpec("k3", "k3 (source Z)", p["k3"], 0.0, 30.0),
            "k4": ParamSpec("k4", "k4 (Y·Z catalysis)", p["k4"], 0.0, 5.0),
            "k5": ParamSpec("k5", "k5 (X² catalysis)", p["k5"], 0.0, 5.0),
            "k6": ParamSpec("k6", "k6 (Y slow/fast)", p["k6"], 0.0, 10.0),
            "k7": ParamSpec("k7", "k7 (X slow/fast)", p["k7"], 0.0, 10.0),
            "k8": ParamSpec("k8", "k8 (X→Z sink)", p["k8"], 0.0, 10.0),
            "k9": ParamSpec("k9", "k9 (Y sink)", p["k9"], 0.0, 5.0),
            "m": ParamSpec("m", "z offset m", p["m"], -5.0, 5.0),
        }

    def default_state(self) -> npt.NDArray[np.float64]:
        return np.array([0.1, 0.1, 0.1], dtype=float)

    def rhs(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        params: Dict[str, float],
    ) -> npt.NDArray[np.float64]:
        X, Y, Z = x
        k1 = params["k1"]
        k2 = params["k2"]
        k3 = params["k3"]
        k4 = params["k4"]
        k5 = params["k5"]
        k6 = params["k6"]
        k7 = params["k7"]
        k8 = params["k8"]
        k9 = params["k9"]
        m = params.get("m", 0.0)

        Z_eff = Z + m

        dX = k1 + k4 * Y * Z_eff - k6 * Y
        dY = k2 + k5 * X**2 - k7 * X - k9 * Y
        dZ = k3 - k8 * X

        return np.array([dX, dY, dZ], dtype=float)

    def rhs_decomposed(
        self,
        t: float,
        x: npt.NDArray[np.float64],
        params: Dict[str, float],
    ) -> Dict[str, npt.NDArray[np.float64]]:
        X, Y, Z = x
        Z_eff = Z + params.get("m", 0.0)

        k1 = params["k1"]
        k2 = params["k2"]
        k3 = params["k3"]
        k4 = params["k4"]
        k5 = params["k5"]
        k6 = params["k6"]
        k7 = params["k7"]
        k8 = params["k8"]
        k9 = params["k9"]

        JS = np.array([k1, k2, k3], dtype=float)
        JC = np.array([k4 * Y * Z_eff, k5 * X**2, 0.0], dtype=float)
        JR = np.array([-k6 * Y, -k7 * X, -k8 * X], dtype=float)
        JE = np.array([0.0, -k9 * Y, 0.0], dtype=float)

        return {"JS": JS, "JC": JC, "JR": JR, "JE": JE}

    def orders(self, params: Dict[str, float]) -> npt.NDArray[np.float64]:
        q = params.get("q", 0.95)
        q = float(min(max(q, 0.01), 1.0))
        return np.array([q, q, q], dtype=float)

    @property
    def state_labels(self) -> list[str]:
        return ["X", "Y", "Z"]
