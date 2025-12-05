from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chemistry", tags=["chemistry"])


class ADRSimulationParams(BaseModel):
    """Request model for ADR bioelectric simulations."""

    t_max: float = Field(500.0, description="Total simulation time")
    dt: float = Field(0.01, description="Time step (used for metadata only)")
    initial_state: Optional[List[float]] = Field(
        None,
        description=(
            "Optional initial state vector for the ADR system. "
            "Length can be the full packed state or Vmem-per-tile (defaults to healthy state when omitted)."
        ),
    )
    mode: str = Field("wound", description="Which experiment to run (currently: 'wound')")


class ADRSimulationResult(BaseModel):
    t: List[float]
    x: List[List[float]]
    lle: Optional[float] = None
    metadata: Dict[str, Any] = {}


def _run_adr_simulation(params: ADRSimulationParams) -> Tuple[List[float], List[List[float]], Optional[float], Dict[str, Any]]:
    """
    Thin wrapper around `sacp_suite.modules.chemistry.adr_bioelectric`.

    Currently uses the wound/repair experiment as a demo. Adjust this to the
    actual API once a general-purpose runner is added.
    """
    try:
        from sacp_suite.modules.chemistry import adr_bioelectric
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to import adr_bioelectric module")
        raise HTTPException(
            status_code=500, detail=f"Chemistry backend unavailable: {exc}"
        ) from exc

    try:
        t_wound = params.t_max / 2.0
        t_final = params.t_max
        adr_params = adr_bioelectric.ADRBioelectricParams()

        user_state = np.asarray(params.initial_state, dtype=float) if params.initial_state else None
        y0 = None
        if user_state is not None:
            expected_full = (3 * adr_bioelectric.N_SITES + 4) * adr_bioelectric.N_TILES
            expected_vmem = adr_bioelectric.N_TILES
            if user_state.size == expected_full:
                y0 = user_state.reshape(-1)
            elif user_state.size == expected_vmem:
                # Map Vmem-per-tile seed onto the full state while keeping defaults for other variables.
                base_state = adr_bioelectric.make_initial_state(adr_params, mode="healthy")
                x, v, r, h1, h4, gamma1, eta = adr_bioelectric.unpack_state(base_state)
                x[0, :] = user_state
                y0 = adr_bioelectric.pack_state(x, v, r, h1, h4, gamma1, eta)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"initial_state length must be either {expected_vmem} (Vmem per tile) "
                        f"or {expected_full} (full packed state); got {user_state.size}"
                    ),
                )

        t_arr, V_traj, _ = adr_bioelectric.run_experiment_wound(
            params=adr_params,
            t_wound=t_wound,
            t_final=t_final,
            y0=y0,
        )
        # V_traj shape: (N_TILES, T); transpose so API returns time-major [T][state]
        x_time_major = np.asarray(V_traj, dtype=float).T.tolist()
        t = [float(v) for v in t_arr]
        meta: Dict[str, Any] = {"mode": params.mode, "dt": params.dt}
        return t, x_time_major, None, meta
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during ADR bioelectric simulation")
        raise HTTPException(
            status_code=500, detail=f"Chemistry simulation failed: {exc}"
        ) from exc


@router.post("/simulate", response_model=ADRSimulationResult)
def simulate_adr(params: ADRSimulationParams) -> ADRSimulationResult:
    """Run an ADR bioelectric simulation and return (t, x, LLE, metadata)."""

    t, x, lle, meta = _run_adr_simulation(params)
    if not t or not x:
        raise HTTPException(
            status_code=500, detail="ADR bioelectric simulation returned no data"
        )

    return ADRSimulationResult(t=t, x=x, lle=lle, metadata=meta)
