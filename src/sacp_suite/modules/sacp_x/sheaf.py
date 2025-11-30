"""Sheaf-style summaries for Lorenz parameter sweeps."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from sacp_suite.core.plugin_api import registry
from sacp_suite.modules.cogmetrics.metrics import estimate_clc_metrics
from sacp_suite.modules.sacp_x.metrics import rosenstein_lle
from sacp_suite.modules.sacp_x import lorenz63 as _lorenz  # noqa: F401  (register plugin)


def _simulate_lorenz(rho: float, sigma: float, beta: float, steps: int, burn_in: int, dt: float) -> np.ndarray:
    dyn = registry.create_dynamics("lorenz63", sigma=sigma, rho=rho, beta=beta)
    state = dyn.default_state().astype(float)
    traj = np.empty((steps, state.shape[0]), dtype=float)
    for i in range(steps):
        state = dyn.rk4_step(state, dt)
        traj[i] = state
    if burn_in > 0:
        traj = traj[burn_in:]
    return traj


def _classify_attractor(lam: float, rms: float) -> str:
    if lam < 0.01:
        return "origin" if rms < 2.0 else "steady_convection"
    if lam < 0.15:
        return "quasi_periodic"
    return "strange_attractor"


def lorenz_attractor_sheaf(
    rhos: Sequence[float],
    *,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    steps: int = 6000,
    burn_in: int = 1000,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """Sweep rho values and organise attractor classes as a crude sheaf."""

    samples: List[Dict[str, Any]] = []
    for rho in sorted(set(rhos)):
        traj = _simulate_lorenz(rho, sigma=sigma, beta=beta, steps=steps, burn_in=burn_in, dt=dt)
        lam = float(rosenstein_lle(traj[:, 0], m=6, tau=4) / max(dt, 1e-9))
        rms = float(np.sqrt(np.mean(np.sum(traj**2, axis=1))))
        attr_class = _classify_attractor(lam, rms)
        clc = estimate_clc_metrics(traj, dt=dt)
        ky_dim = 0.0
        if attr_class == "steady_convection":
            ky_dim = 1.0
        elif attr_class == "quasi_periodic":
            ky_dim = 1.5
        elif attr_class == "strange_attractor":
            ky_dim = 2.0 + min(0.6, lam * 0.1)

        samples.append(
            {
                "rho": float(rho),
                "class_label": attr_class,
                "lambda_max": lam,
                "ky_dim": float(ky_dim),
                "rms": rms,
                "tau_past": clc["tau_past"],
                "tau_future": clc["tau_future"],
                "capture": clc["capture"],
                "clc": clc["clc"],
            }
        )

    sections: List[Dict[str, Any]] = []
    obstructions: List[Dict[str, Any]] = []
    if samples:
        start = samples[0]["rho"]
        current_class = samples[0]["class_label"]
        for prev, nxt in zip(samples, samples[1:]):
            if nxt["class_label"] != current_class:
                sections.append({"start": start, "end": prev["rho"], "class_label": current_class, "persist": True})
                obstructions.append(
                    {
                        "left": prev["rho"],
                        "right": nxt["rho"],
                        "reason": f"class shift {current_class}→{nxt['class_label']}",
                    }
                )
                start = nxt["rho"]
                current_class = nxt["class_label"]
        sections.append({"start": start, "end": samples[-1]["rho"], "class_label": current_class, "persist": True})

    return {"samples": samples, "sections": sections, "obstructions": obstructions}
