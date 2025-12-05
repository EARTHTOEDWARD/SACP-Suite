"""Bouquet stack API endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, conlist

from sacp_suite.modules.bouquet.core import (
    SelfTuningConfig,
    analyze_clc_vs_bouquet,
    build_bouquet_geom,
    run_bouquet,
    run_bouquet_tuning,
)

router = APIRouter(prefix="/bouquet", tags=["bouquet"])


class BouquetRunRequest(BaseModel):
    alpha: float = Field(1.3, description="Scale factor per layer")
    n_layers: int = Field(4, ge=1, le=24)
    N: int = Field(12, ge=4, le=128, description="Sites per layer")
    dt: float = Field(0.01, gt=0)
    n_steps: int = Field(2000, ge=1)
    log_every: int = Field(10, ge=1)
    k_vert: float = Field(0.05, ge=0.0)
    seed: int = Field(123, description="Base RNG seed")
    self_tune: bool = Field(False, description="Enable vertical self-tuning gate")
    target_S_ring: float = Field(1.5, ge=0.0)
    control_every: int = Field(200, ge=1)
    control_window: int = Field(40, ge=2)


class BouquetScanRequest(BaseModel):
    alphas: List[float] = Field(..., min_length=1, description="List of alpha values to scan")
    n_layers: int = Field(4, ge=1, le=24)
    N: int = Field(10, ge=4, le=128)
    n_steps: int = Field(1500, ge=1)
    log_every: int = Field(10, ge=1)
    dt: float = Field(0.01, gt=0)
    k_vert: float = Field(0.05, ge=0.0)
    seed: int = Field(123, description="Base RNG seed")


@router.post("/run")
def run(req: BouquetRunRequest) -> dict:
    """
    Run a single bouquet-stack simulation.
    """
    if req.self_tune:
        tuning = SelfTuningConfig(target_S_ring=req.target_S_ring)
        result = run_bouquet_tuning(
            alpha=req.alpha,
            n_layers=req.n_layers,
            N=req.N,
            dt=req.dt,
            n_steps=req.n_steps,
            log_every=req.log_every,
            k_vert=req.k_vert,
            seed=req.seed,
            control_every=req.control_every,
            control_window=req.control_window,
            tuning=tuning,
        )
        result["mode"] = "self_tuned"
        return result

    result = run_bouquet(
        alpha=req.alpha,
        n_layers=req.n_layers,
        N=req.N,
        dt=req.dt,
        n_steps=req.n_steps,
        log_every=req.log_every,
        k_vert=req.k_vert,
        seed=req.seed,
    )
    result["mode"] = "static"
    return result


@router.post("/scan")
def scan(req: BouquetScanRequest) -> dict:
    """
    Scan multiple alpha values and return bouquet metrics per alpha.
    """
    outputs: List[dict] = []
    for alpha in req.alphas:
        geom = build_bouquet_geom(n_layers=req.n_layers, alpha=alpha)
        # run_bouquet already builds geom internally; reuse to avoid drift.
        res = run_bouquet(
            alpha=alpha,
            n_layers=req.n_layers,
            N=req.N,
            dt=req.dt,
            n_steps=req.n_steps,
            log_every=req.log_every,
            k_vert=req.k_vert,
            seed=req.seed,
        )
        outputs.append({"alpha": alpha, **res})

    return {"alphas": req.alphas, "results": outputs}
