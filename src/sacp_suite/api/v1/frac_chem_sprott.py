from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sacp_suite.abtc import (
    FracChemSprottDynamics,
    basin_scan,
    bifurcation_scan,
    compute_complexity_grid,
    simulate_fde,
)

router = APIRouter(prefix="/frac-chem-sprott", tags=["frac_chem_sprott"])

dyn = FracChemSprottDynamics()


class FracChemSprottSimRequest(BaseModel):
    params: Dict[str, float] = Field(default_factory=dyn.default_params)
    x0: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    t_max: float = 200.0
    dt: float = 0.002
    t_transient: float = 40.0
    sample_stride: int = 5
    return_currents: bool = True
    return_0_1_test: bool = False
    return_complexity: bool = False
    observable: Literal["x", "y", "z"] = "x"


class TrajectorySeries(BaseModel):
    t: List[float]
    x: List[float]
    y: List[float]
    z: List[float]


class CurrentsSeries(BaseModel):
    JS: List[List[float]]
    JC: List[List[float]]
    JR: List[List[float]]
    JE: List[List[float]]


class FracChemSprottSimResponse(BaseModel):
    traj: TrajectorySeries
    params: Dict[str, float]
    lle: Optional[float] = None
    k01: Optional[float] = None
    complexity: Optional[Dict[str, float]] = None
    currents: Optional[CurrentsSeries] = None
    diverged: Optional[bool] = None


class BifurcationScanRequest(BaseModel):
    scan_param: Literal["q", "k1"]
    start: float
    stop: float
    num: int = 90
    observable: Literal["x", "y", "z"] = "x"
    params: Dict[str, float] = Field(default_factory=dyn.default_params)
    t_max: float = 120.0
    dt: float = 0.002
    t_transient: float = 30.0
    sample_stride: int = 10


class BifurcationScanResponse(BaseModel):
    param_values: List[float]
    observable_samples: List[List[float]]


class ComplexityGridRequest(BaseModel):
    q_min: float = 0.55
    q_max: float = 1.0
    q_steps: int = 12
    k1_min: float = 0.0
    k1_max: float = 0.07
    k1_steps: int = 12
    observable: Literal["x", "y", "z"] = "x"
    params: Dict[str, float] = Field(default_factory=dyn.default_params)
    # TODO: add async/background job path for high-resolution grids in production.


class ComplexityGridResponse(BaseModel):
    q_values: List[float]
    k1_values: List[float]
    se_grid: List[List[float]]
    c0_grid: List[List[float]]


class BasinRequest(BaseModel):
    x_min: float = 0.0
    x_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 5.0
    nx: int = 200
    nz: int = 200
    y0: float = 0.1
    params: Dict[str, float] = Field(default_factory=dyn.default_params)
    t_max: float = 500.0
    dt: float = 0.01
    t_transient: float = 150.0


class BasinResponse(BaseModel):
    x_values: List[float]
    z_values: List[float]
    basin_labels: List[List[int]]


@router.post("/simulate", response_model=FracChemSprottSimResponse)
def simulate_frac_chem_sprott(req: FracChemSprottSimRequest) -> FracChemSprottSimResponse:
    params = dyn.default_params()
    params.update(req.params or {})
    x0 = np.array(req.x0, dtype=float)

    try:
        traj, lle, extras = simulate_fde(
            dynamics=dyn,
            params=params,
            x0=x0,
            t_max=req.t_max,
            dt=req.dt,
            t_transient=req.t_transient,
            sample_stride=req.sample_stride,
            compute_0_1=req.return_0_1_test,
            compute_complexity=req.return_complexity,
            observable=req.observable,
            return_currents=req.return_currents,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc

    currents = extras.get("currents") if isinstance(extras, dict) else None
    return FracChemSprottSimResponse(
        traj=TrajectorySeries(
            t=traj["t"].tolist(),
            x=traj["x"].tolist(),
            y=traj["y"].tolist(),
            z=traj["z"].tolist(),
        ),
        params=params,
        lle=lle,
        k01=extras.get("k01") if isinstance(extras, dict) else None,
        complexity=extras.get("complexity") if isinstance(extras, dict) else None,
        diverged=extras.get("diverged") if isinstance(extras, dict) else None,
        # currents arrays are already finite after nan_to_num in engine
        currents=(
            CurrentsSeries(
                JS=currents["JS"].tolist(),
                JC=currents["JC"].tolist(),
                JR=currents["JR"].tolist(),
                JE=currents["JE"].tolist(),
            )
            if currents is not None
            else None
        ),
    )


@router.post("/bifurcation", response_model=BifurcationScanResponse)
def frac_chem_sprott_bifurcation(req: BifurcationScanRequest) -> BifurcationScanResponse:
    params = dyn.default_params()
    params.update(req.params or {})
    try:
        result = bifurcation_scan(
            dynamics=dyn,
            params=params,
            scan_param=req.scan_param,
            start=req.start,
            stop=req.stop,
            num=req.num,
            observable=req.observable,
            t_max=req.t_max,
            dt=req.dt,
            t_transient=req.t_transient,
            sample_stride=req.sample_stride,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Bifurcation scan failed: {exc}") from exc

    return BifurcationScanResponse(
        param_values=result["param_values"],
        observable_samples=[np.asarray(row, dtype=float).tolist() for row in result["samples"]],
    )


@router.post("/complexity-grid", response_model=ComplexityGridResponse)
def frac_chem_sprott_complexity_grid(req: ComplexityGridRequest) -> ComplexityGridResponse:
    params = dyn.default_params()
    params.update(req.params or {})
    try:
        grid = compute_complexity_grid(
            dynamics=dyn,
            base_params=params,
            q_min=req.q_min,
            q_max=req.q_max,
            q_steps=req.q_steps,
            k1_min=req.k1_min,
            k1_max=req.k1_max,
            k1_steps=req.k1_steps,
            observable=req.observable,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Complexity grid failed: {exc}") from exc

    return ComplexityGridResponse(
        q_values=grid["q_values"],
        k1_values=grid["k1_values"],
        se_grid=np.asarray(grid["se"], dtype=float).tolist(),
        c0_grid=np.asarray(grid["c0"], dtype=float).tolist(),
    )


@router.post("/basin", response_model=BasinResponse)
def frac_chem_sprott_basin(req: BasinRequest) -> BasinResponse:
    params = dyn.default_params()
    params.update(req.params or {})
    try:
        basins = basin_scan(
            dynamics=dyn,
            params=params,
            x_min=req.x_min,
            x_max=req.x_max,
            z_min=req.z_min,
            z_max=req.z_max,
            nx=req.nx,
            nz=req.nz,
            y0=req.y0,
            t_max=req.t_max,
            dt=req.dt,
            t_transient=req.t_transient,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Basin scan failed: {exc}") from exc

    labels = np.asarray(basins["labels"], dtype=int)
    return BasinResponse(
        x_values=basins["x_values"],
        z_values=basins["z_values"],
        basin_labels=labels.tolist(),
    )
