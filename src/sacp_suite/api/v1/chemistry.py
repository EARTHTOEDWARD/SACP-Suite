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
    initial_state: List[float] = Field(
        ..., description="Initial state vector for the ADR system (optional; length must match model)"
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
        t_arr, V_traj, _ = adr_bioelectric.run_experiment_wound(
            params=adr_bioelectric.ADRBioelectricParams(),
            t_wound=t_wound,
            t_final=t_final,
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
