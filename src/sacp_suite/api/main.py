"""FastAPI application that unifies SACP Suite services."""

from __future__ import annotations

import os
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sacp_suite.core.plugin_api import registry
from sacp_suite.modules.sacp_x import lorenz63 as _lorenz  # noqa: F401  (register plugin)
from sacp_suite.modules.sacp_x.metrics import rosenstein_lle
from sacp_suite.modules.attractorhedron.operator import (
    build_ulam,
    analyze_operator,
    left_right_gate,
)


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


app = FastAPI(title="SACP Suite API", version="0.1.0")

origins = [o.strip() for o in os.getenv("SACP_CORS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class SimRequest(BaseModel):
    model: str = Field(default="lorenz63")
    params: dict = Field(default_factory=dict)
    T: float = 50.0
    dt: float = 0.01


class SimResult(BaseModel):
    time: List[float]
    trajectory: List[List[float]]
    labels: List[str]


class LLERequest(BaseModel):
    series: List[float]
    m: int = 6
    tau: int = 4


class OperatorBuildRequest(BaseModel):
    points: List[List[float]]
    nx: int = 25
    ny: int = 25


class OperatorAnalyzeRequest(BaseModel):
    P: List[List[float]]
    mean_return: float = 1.0
    nx: int = 25
    ny: int = 25


class GateSweepRequest(BaseModel):
    P: List[List[float]]
    nx: int = 25
    ny: int = 25
    alphas: List[float] = Field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])


@app.get("/health")
def health():
    return {"ok": True, "models": registry.list_dynamics()}


@app.post("/simulate", response_model=SimResult)
def simulate(req: SimRequest):
    dyn = registry.create_dynamics(req.model, **req.params)
    traj = dyn.simulate(T=req.T, dt=req.dt)
    t = [i * req.dt for i in range(traj.shape[0])]
    return {"time": t, "trajectory": traj.tolist(), "labels": dyn.state_labels}


@app.post("/metrics/lle")
def metrics_lle(req: LLERequest):
    return {"lle": rosenstein_lle(np.array(req.series), m=req.m, tau=req.tau)}


@app.post("/operator/build")
def operator_build(req: OperatorBuildRequest):
    P, bounds = build_ulam(np.array(req.points), nx=req.nx, ny=req.ny)
    return {"P": P.tolist(), "bounds": [float(b) for b in bounds]}


@app.post("/operator/analyze")
def operator_analyze(req: OperatorAnalyzeRequest):
    P = np.array(req.P, dtype=float)
    return analyze_operator(P, mean_return=req.mean_return, nx=req.nx, ny=req.ny)


@app.post("/gate/sweep")
def gate_sweep(req: GateSweepRequest):
    P = np.array(req.P, dtype=float)
    out = []
    for alpha in req.alphas:
        Pg = left_right_gate(P, nx=req.nx, ny=req.ny, alpha=alpha)
        res = analyze_operator(Pg, mean_return=1.0, nx=req.nx, ny=req.ny)
        res.update({"alpha": alpha})
        out.append(res)
    return {"results": out}


def run() -> None:
    import uvicorn

    host = os.getenv("SACP_BIND", "127.0.0.1")
    port = int(os.getenv("SACP_PORT", "8000"))
    uvicorn.run("sacp_suite.api.main:app", host=host, port=port, reload=_bool_env("SACP_RELOAD", True))
