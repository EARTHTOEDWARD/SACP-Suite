from __future__ import annotations

"""Additional strange attractors (equations + defaults) inspired by the Shashank Tomar gallery."""

from typing import Any, Dict, List

import numpy as np

from sacp_suite.core.plugin_api import BaseDynamics, registry


def _np(arr: List[float]) -> np.ndarray:
    return np.array(arr, dtype=float)


def rossler_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    return np.array([-y - z, x + p["a"] * y, p["b"] + z * (x - p["c"])], dtype=float)


def chen_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    return np.array(
        [
            p["a"] * (y - x),
            (p["c"] - p["a"]) * x - x * z + p["c"] * y,
            x * y - p["b"] * z,
        ],
        dtype=float,
    )


def aizawa_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    dx = (z - p["b"]) * x - p["d"] * y
    dy = p["d"] * x + (z - p["b"]) * y
    dz = (
        p["c"]
        + p["a"] * z
        - (z ** 3) / 3.0
        - (x**2 + y**2) * (1 + p["e"] * z)
        + p["f"] * z * x**3
    )
    return np.array([dx, dy, dz], dtype=float)


def thomas_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    b = p["b"]
    return np.array([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z], dtype=float)


def dadras_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    return np.array(
        [
            y - p["a"] * x + p["b"] * y * z,
            p["c"] * y - x * z + z,
            p["d"] * x * y - p["e"] * z,
        ],
        dtype=float,
    )


def halvorsen_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    a = p["a"]
    return np.array(
        [
            -a * x - 4 * y - 4 * z - y * y,
            -a * y - 4 * z - 4 * x - z * z,
            -a * z - 4 * x - 4 * y - x * x,
        ],
        dtype=float,
    )


def rabfab_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    return np.array(
        [
            y * (z - 1 + x * x) + p["a"] * x,
            x * (3 * z + 1 - x * x) + p["a"] * y,
            -2 * z * (p["b"] + x * y),
        ],
        dtype=float,
    )


def sprott_a_deriv(s: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    x, y, z = s
    a = p["a"]
    return np.array([y, z, -x - a * y - z + y * y], dtype=float)


ATTRACTOR_LIBRARY: List[Dict[str, Any]] = [
    {
        "key": "rossler",
        "name": "Rössler",
        "default_state": [0.1, 0.1, 0.0],
        "derivative": rossler_deriv,
        "params": {
            "a": {"default": 0.2, "min": 0.0, "max": 0.4, "step": 0.01},
            "b": {"default": 0.2, "min": 0.0, "max": 1.5, "step": 0.01},
            "c": {"default": 5.7, "min": 2.0, "max": 12.0, "step": 0.1},
        },
    },
    {
        "key": "chen",
        "name": "Chen",
        "default_state": [0.1, 0.1, 0.1],
        "derivative": chen_deriv,
        "params": {
            "a": {"default": 35.0, "min": 5.0, "max": 40.0, "step": 0.5},
            "b": {"default": 3.0, "min": 0.1, "max": 5.0, "step": 0.1},
            "c": {"default": 28.0, "min": 10.0, "max": 40.0, "step": 0.5},
        },
    },
    {
        "key": "aizawa",
        "name": "Aizawa",
        "default_state": [0.1, 0.0, 0.0],
        "derivative": aizawa_deriv,
        "params": {
            "a": {"default": 0.95, "min": 0.5, "max": 1.2, "step": 0.01},
            "b": {"default": 0.7, "min": 0.2, "max": 1.5, "step": 0.01},
            "c": {"default": 0.6, "min": 0.2, "max": 1.2, "step": 0.01},
            "d": {"default": 3.5, "min": 1.0, "max": 6.0, "step": 0.1},
            "e": {"default": 0.25, "min": 0.0, "max": 0.6, "step": 0.01},
            "f": {"default": 0.1, "min": 0.0, "max": 0.4, "step": 0.01},
        },
    },
    {
        "key": "thomas",
        "name": "Thomas",
        "default_state": [0.1, 0.1, 0.1],
        "derivative": thomas_deriv,
        "params": {"b": {"default": 0.208186, "min": 0.05, "max": 0.4, "step": 0.005}},
    },
    {
        "key": "dadras",
        "name": "Dadras",
        "default_state": [1.0, 1.0, 1.0],
        "derivative": dadras_deriv,
        "params": {
            "a": {"default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "b": {"default": 2.7, "min": 0.0, "max": 5.0, "step": 0.1},
            "c": {"default": 1.7, "min": 0.0, "max": 3.5, "step": 0.05},
            "d": {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.05},
            "e": {"default": 9.0, "min": 0.0, "max": 12.0, "step": 0.1},
        },
    },
    {
        "key": "halvorsen",
        "name": "Halvorsen",
        "default_state": [1.0, 0.0, 0.0],
        "derivative": halvorsen_deriv,
        "params": {"a": {"default": 1.4, "min": 0.5, "max": 2.5, "step": 0.05}},
    },
    {
        "key": "rabfab",
        "name": "Rabinovich–Fabrikant",
        "default_state": [1.0, 1.0, 1.0],
        "derivative": rabfab_deriv,
        "params": {
            "a": {"default": 0.14, "min": 0.05, "max": 0.5, "step": 0.01},
            "b": {"default": 0.10, "min": 0.05, "max": 0.3, "step": 0.005},
        },
    },
    {
        "key": "sprott_a",
        "name": "Sprott A",
        "default_state": [0.0, 1.0, 0.0],
        "derivative": sprott_a_deriv,
        "params": {"a": {"default": 0.2, "min": 0.05, "max": 1.0, "step": 0.01}},
    },
]


def _register_from_library() -> List[Dict[str, Any]]:
    """Register all attractors in the plugin registry and return public metadata."""
    public_meta: List[Dict[str, Any]] = []
    for item in ATTRACTOR_LIBRARY:
        key = item["key"]
        defaults = {k: v["default"] for k, v in item["params"].items()}
        default_state = _np(item.get("default_state", [1.0, 1.0, 1.0]))
        deriv_fn = item["derivative"]
        name = item["name"]

        # Build a concrete subclass bound to the derivative + defaults.
        def make_cls(k: str, d: Dict[str, float], ds: np.ndarray, fn, human: str):
            def get_default_params(self) -> Dict[str, float]:  # type: ignore[override]
                return d.copy()

            def default_state(self) -> np.ndarray:  # type: ignore[override]
                return ds.copy()

            def derivative(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:  # type: ignore[override]
                return fn(state, self.params)

            @property  # type: ignore[misc]
            def name(self) -> str:
                return human

            @property  # type: ignore[misc]
            def state_labels(self) -> List[str]:
                return ["X", "Y", "Z"]

            return type(
                f"{human.replace(' ', '')}Dynamics",
                (BaseDynamics,),
                {
                    "get_default_params": get_default_params,
                    "default_state": default_state,
                    "derivative": derivative,
                    "name": name,
                    "state_labels": state_labels,
                },
            )

        cls = make_cls(key, defaults, default_state, deriv_fn, name)
        registry.register_dynamics(key, cls)

        public_meta.append(
            {
                "key": key,
                "name": name,
                "params": item["params"],
                "default_state": item.get("default_state", [1.0, 1.0, 1.0]),
                "state_labels": ["X", "Y", "Z"],
            }
        )
    return public_meta


PUBLIC_ATTRACTORS = _register_from_library()


def list_attractors() -> List[Dict[str, Any]]:
    return PUBLIC_ATTRACTORS
