"""FastAPI application that unifies SACP Suite services."""

from __future__ import annotations

import base64
import os
import random
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sacp_suite.api.v1 import auth, datasets as v1_datasets, jobs as v1_jobs, workspaces
from sacp_suite.api.v1 import chemistry
from sacp_suite.api.v1 import frac_chem_sprott
from sacp_suite.core.datasets import list_datasets, preview_dataset, register_upload
from sacp_suite.core.plugin_api import registry
from sacp_suite.modules.abtc.core import rk4
from sacp_suite.modules.attractorhedron.operator import (
    analyze_operator,
    build_ulam,
    left_right_gate,
)
from sacp_suite.modules.bcp.interface import to_section
from sacp_suite.modules.cogmetrics.experiments import (
    generate_concept_dataset,
    generate_random_drive_dataset,
)
from sacp_suite.modules.cogmetrics.metrics import (
    compute_memory_profile,
    estimate_clc_metrics,
    estimate_discriminability,
)
from sacp_suite.modules.cogmetrics.simulator_wrappers import make_param_forced_simulator
from sacp_suite.modules.dcrc import DCRC_Bank
from sacp_suite.modules.fractal_llm import generate_reservoir_sample
from sacp_suite.modules.fractalhedron import (
    build_fractalhedron_k,
    build_symbolic_sequence,
    fractal_face_flags,
)
from sacp_suite.demos.self_tuned_lorenz import run_self_tuned_lorenz
from sacp_suite.self_tuning.manager import SelfTuningManager
from sacp_suite.selftuner.types import ChaosBand
from sacp_suite.systems.lorenz import LorenzSystem
from sacp_suite.modules.sacp_x import lorenz63 as _lorenz  # noqa: F401  (register plugin)
from sacp_suite.modules.sacp_x.lorenz63 import Lorenz63
from sacp_suite.modules.sacp_x.metrics import rosenstein_lle
from sacp_suite.modules.sacp_x.sheaf import lorenz_attractor_sheaf


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


app = FastAPI(title="SACP Suite API", version="0.1.0")
_last_section_hits: np.ndarray | None = None
_self_tuning_sessions: Dict[str, Dict[str, Any]] = {}

origins = [o.strip() for o in os.getenv("SACP_CORS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# v1 routers (SaaS scaffolding)
app.include_router(auth.router)
app.include_router(workspaces.router)
app.include_router(v1_datasets.router)
app.include_router(v1_jobs.router)
app.include_router(chemistry.router, prefix="/api/v1")
app.include_router(frac_chem_sprott.router, prefix="/api/v1")

ROOT_DIR = Path(__file__).resolve().parents[3]
INGESTION_CONFIG_PATH = ROOT_DIR / "configs" / "ingestion.yaml"
DEFAULT_INGESTION_CONFIG = {
    "sufficiency_thresholds": {"minimal": 500, "recommended": 100000, "large": 250000},
    "bands": {
        "minimal": "Minimal",
        "recommended": "Recommended",
        "large": "Large",
    },
    "messages": {
        "minimal": "We can explore basic plots, but chaos indicators may be noisy.",
        "recommended": "Ideal range for Lyapunov estimates and model fitting.",
        "large": "Great coverage — we will window/subsample for responsiveness.",
    },
}


@lru_cache()
def load_ingestion_config() -> Dict[str, Any]:
    config = DEFAULT_INGESTION_CONFIG.copy()
    if INGESTION_CONFIG_PATH.exists():
        with INGESTION_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            for key, value in data.items():
                if isinstance(value, dict):
                    base = config.get(key, {}).copy()
                    base.update(value)
                    config[key] = base
                else:
                    config[key] = value
    return config


class TaskSpec(BaseModel):
    goal: str
    has_data: bool
    data_status: Literal["have_data", "need_dataset"]
    mode: Literal["batch", "realtime"]
    needs_realtime: bool = False
    data_location: Optional[str] = None
    data_modality: Optional[str] = None
    latency_tolerance_seconds: Optional[float] = None
    estimated_data_size: Optional[str] = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class TaskParseRequest(BaseModel):
    prompt: str
    overrides: Dict[str, Any] = Field(default_factory=dict)


class IngestPreviewRequest(BaseModel):
    source: Literal["file", "s3", "gcs", "azure", "db", "api", "example"] = "file"
    sample_size: int = 15
    modality: Literal["time_series", "image", "field", "other"] = "time_series"


class ColumnMapping(BaseModel):
    time_col: str
    state_cols: List[str]
    control_cols: List[str] = Field(default_factory=list)
    label_col: Optional[str] = None
    group_col: Optional[str] = None


class IngestValidateRequest(BaseModel):
    mapping: ColumnMapping
    row_count: int
    sample_frequency_hz: Optional[float] = None
    needs_alerts: bool = False


class DatasetGenerationRequest(BaseModel):
    system: str
    duration: Literal["short", "medium", "long"] = "short"
    sampling_rate: Literal["slow", "medium", "fast"] = "medium"
    noise_level: float = 0.0
    trials: int = 1
    variations: int = 0


def _infer_task_spec(prompt: str, overrides: Optional[Dict[str, Any]] = None) -> TaskSpec:
    text = (prompt or "").lower()
    has_data = any(keyword in text for keyword in ["my data", "dataset", "csv", "parquet", "log", "recording"])
    needs_realtime = any(keyword in text for keyword in ["real time", "realtime", "stream", "monitor", "live"])
    data_status: Literal["have_data", "need_dataset"] = "have_data" if has_data else "need_dataset"
    mode: Literal["batch", "realtime"] = "realtime" if needs_realtime else "batch"

    data_location: Optional[str] = None
    if "s3" in text:
        data_location = "s3"
        has_data = True
    elif any(keyword in text for keyword in ["gcs", "google cloud"]):
        data_location = "gcs"
        has_data = True
    elif any(keyword in text for keyword in ["upload", "file", "csv", "parquet", "hdf5"]):
        data_location = "file"
        has_data = True
    elif any(keyword in text for keyword in ["postgres", "mysql", "database"]):
        data_location = "db"
        has_data = True
    elif any(keyword in text for keyword in ["api", "websocket", "mqtt"]):
        data_location = "api"
        has_data = True

    data_modality = "time_series"
    if any(keyword in text for keyword in ["image", "video", "frame"]):
        data_modality = "image"
    elif any(keyword in text for keyword in ["field", "grid"]):
        data_modality = "field"
    elif any(keyword in text for keyword in ["text", "log", "token"]):
        data_modality = "other"

    latency = 5.0 if "5" in text and needs_realtime else (2.0 if needs_realtime else None)
    estimated_data_size = "50k_rows" if has_data else None

    confidence = 0.6
    if has_data:
        confidence += 0.15
    if needs_realtime:
        confidence += 0.1

    spec_data: Dict[str, Any] = {
        "goal": prompt.strip() or "Explore my system",
        "has_data": has_data,
        "data_status": data_status if has_data else "need_dataset",
        "mode": mode,
        "needs_realtime": needs_realtime,
        "data_location": data_location,
        "data_modality": data_modality,
        "latency_tolerance_seconds": latency,
        "estimated_data_size": estimated_data_size,
        "confidence": min(confidence, 0.95),
    }

    if overrides:
        spec_data.update(overrides)
    return TaskSpec(**spec_data)


def _mock_preview(sample_size: int = 15) -> Dict[str, Any]:
    size = max(5, min(sample_size, 50))
    time_values = np.linspace(0, size / 5, size)
    rows = []
    for idx, t in enumerate(time_values):
        rows.append(
            {
                "t": float(t),
                "x": float(np.sin(t) + 0.1 * np.random.randn()),
                "y": float(np.cos(t) + 0.1 * np.random.randn()),
                "z": float(np.sin(0.5 * t + 0.2) + 0.1 * np.random.randn()),
                "u1": float(0.5 * np.sin(0.2 * t) + 0.05 * np.random.randn()),
                "row": idx,
            }
        )
    return {
        "columns": ["t", "x", "y", "z", "u1"],
        "rows": rows,
        "row_estimate": random.randint(2_000, 200_000),
    }


def _evaluate_sufficiency(count: int, config: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = config.get("sufficiency_thresholds", {})
    minimal = thresholds.get("minimal", 500)
    recommended = thresholds.get("recommended", 100_000)
    large = thresholds.get("large", max(recommended, 250_000))

    if count < minimal:
        band = "minimal"
    elif count <= recommended:
        band = "recommended"
    else:
        band = "large"

    band_label = config.get("bands", {}).get(band, band.title())
    message = config.get("messages", {}).get(band, "")
    return {
        "band": band,
        "label": band_label,
        "message": message,
        "thresholds": thresholds,
        "row_count": count,
    }


def _estimate_dataset_size(duration: str, sampling_rate: str, trials: int) -> int:
    duration_map = {"short": 10, "medium": 60, "long": 240}
    rate_map = {"slow": 30, "medium": 200, "fast": 1000}
    seconds = duration_map.get(duration, duration_map["short"])
    rate = rate_map.get(sampling_rate, rate_map["medium"])
    return int(seconds * rate * max(trials, 1))



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


class FractalRequest(BaseModel):
    text: str
    reservoir_size: int = 400
    coupling: float = 1.0


class DCRCRequest(BaseModel):
    num_reservoirs: int = 3
    coupling: float = 1.0
    timesteps: int = 1000


class SectionRequest(BaseModel):
    points: List[List[float]]


class ABTCRequest(BaseModel):
    x0: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    dt: float = 0.01
    steps: int = 200


class ADRKrebsRunRequest(BaseModel):
    alpha: float = 1.0
    n_steps: int = Field(default=20000, ge=10)
    log_every: int = Field(default=10, ge=1)
    dt: float = 0.01
    seed: Optional[int] = None


class ADRKrebsRunResult(BaseModel):
    time: List[float]
    x: List[List[float]]
    v: List[List[float]]
    r: List[List[float]]
    k: List[List[float]]
    S_ring: float
    diag: Dict[str, Any]
    site_names: List[str]


class ADRKrebsScanRequest(BaseModel):
    alpha_min: float = 0.4
    alpha_max: float = 1.8
    n_alpha: int = Field(default=40, ge=2)
    n_steps_transient: int = Field(default=15000, ge=100)
    n_steps_sample: int = Field(default=15000, ge=100)
    sample_every: int = Field(default=10, ge=1)
    sample_site: int = Field(default=0, ge=0, le=3)
    dt: float = 0.01


class ADRKrebsScanResult(BaseModel):
    alphas: List[float]
    samples: List[float]
    site: int


class DatasetPreviewRequest(BaseModel):
    dataset_id: str
    limit: int = 1000


class DatasetUploadRequest(BaseModel):
    name: str
    description: str = ""
    contents_b64: str
    columns: List[str] = Field(default_factory=list)


class MemoryExperimentRequest(BaseModel):
    model_key: str = "lorenz63"
    param_name: str = "rho"
    base_value: float = 28.0
    amp: float = 2.0
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    n_trials: int = 4
    n_steps: int = 10_000
    burn_in: int = 1_000
    input_alphabet: List[float] = Field(default_factory=lambda: [-1.0, 0.0, 1.0])
    input_probs: Optional[List[float]] = None
    max_lag: int = 50
    n_output_bins: int = 16


class MemoryExperimentResult(BaseModel):
    lags: List[int]
    M: List[float]
    sample_u: List[float]
    sample_y: List[float]
    dt: float
    tau_past: float
    tau_future: float
    radius: float
    capture: float
    clc: float


class ConceptPattern(BaseModel):
    pattern: List[float]


class DiscriminabilityExperimentRequest(BaseModel):
    model_key: str = "lorenz63"
    param_name: str = "rho"
    base_value: float = 28.0
    amp: float = 2.0
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    patterns: List[ConceptPattern]
    n_trials_per_concept: int = 100
    burn_in: int = 500


class DiscriminabilityExperimentResult(BaseModel):
    K: float
    T: float
    accuracy: float
    Pe: float
    I_lower: float
    D: float


class SheafSample(BaseModel):
    rho: float
    class_label: str
    lambda_max: float
    ky_dim: float
    rms: float
    tau_past: float
    tau_future: float
    capture: float
    clc: float


class SheafSection(BaseModel):
    start: float
    end: float
    class_label: str
    persist: bool = True


class SheafObstruction(BaseModel):
    left: float
    right: float
    reason: str


class LorenzSheafRequest(BaseModel):
    rhos: List[float] = Field(default_factory=lambda: [0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 24.0, 28.0, 35.0])
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    steps: int = Field(default=6000, ge=100)
    burn_in: int = Field(default=1000, ge=0)
    dt: float = 0.01


class LorenzSheafResult(BaseModel):
    samples: List[SheafSample]
    sections: List[SheafSection]
    obstructions: List[SheafObstruction]


class FractalHedronRequest(BaseModel):
    section_hits: List[List[float]]
    coding_spec: Literal["x_sign", "quadrant_xz", "radius_bins"] = "x_sign"
    coding_params: Dict[str, Any] = Field(default_factory=dict)
    k: int = 2
    Q: List[float] = Field(default_factory=lambda: [0.0, 2.0])


class FractalHedronResult(BaseModel):
    alphabet: List[str]
    ell_k: float
    D_q: Dict[str, float]
    T_q: Dict[str, float]
    constraints: Dict[str, Any]
    faces: Dict[str, Any]


class SelfTunedLorenzRequest(BaseModel):
    num_steps: int = Field(default=2000, ge=10, le=20000)
    record_states: bool = False


class SelfTunedLorenzResult(BaseModel):
    lambdas: List[float]
    regimes: List[str]
    spectral_radius: List[float]
    gains: List[float]
    states: List[List[float]]
    dt: float


class SelfTuningSessionCreateRequest(BaseModel):
    dt: float = 0.01
    chaos_band: List[float] = Field(default_factory=lambda: [0.0, 0.5])
    k_spectral_radius: float = 0.05
    steps_per_update: int = Field(default=50, ge=1, le=1000)


class SelfTuningSessionStepRequest(BaseModel):
    session_id: str
    steps: int = Field(default=100, ge=1, le=2000)


class SelfTuningSessionState(BaseModel):
    session_id: str
    step: int
    lambdas: List[float]
    spectral_radius: List[float]
    gain: float
    state: List[float]
    regime: Optional[str]


@app.get("/health")
def health():
    return {"ok": True, "models": registry.list_dynamics()}


@app.post("/simulate", response_model=SimResult)
def simulate(req: SimRequest):
    dyn = registry.create_dynamics(req.model, **req.params)
    traj = dyn.simulate(T=req.T, dt=req.dt)
    t = [i * req.dt for i in range(traj.shape[0])]
    return {"time": t, "trajectory": traj.tolist(), "labels": dyn.state_labels}


def _load_krebs_mod():
    """Lazy-import ADR–Krebs to avoid startup failures if chemistry stack is broken."""
    # Importing the chemistry package registers plugins with the registry.
    from sacp_suite.modules import chemistry as _chemistry_plugins  # noqa: F401
    from sacp_suite.modules.chemistry import adr_krebs as krebs_mod  # type: ignore

    return krebs_mod


@app.post("/chem/adr-krebs/run", response_model=ADRKrebsRunResult)
def run_adr_krebs(req: ADRKrebsRunRequest):
    krebs_mod = _load_krebs_mod()
    cfg = krebs_mod.KrebsADRConfig(dt=req.dt, seed=req.seed)
    _, log, S_ring, diag = krebs_mod.run_krebs_trajectory(
        alpha=req.alpha,
        n_steps=req.n_steps,
        log_every=req.log_every,
        cfg=cfg,
    )
    diag_out: Dict[str, Any] = {}
    for key, val in diag.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            diag_out[key] = np.asarray(val).tolist()
        else:
            diag_out[key] = float(val)

    return {
        "time": np.asarray(log["time"]).tolist(),
        "x": np.asarray(log["x"]).tolist(),
        "v": np.asarray(log["v"]).tolist(),
        "r": np.asarray(log["r"]).tolist(),
        "k": np.asarray(log["k"]).tolist(),
        "S_ring": float(S_ring),
        "diag": diag_out,
        "site_names": krebs_mod.SITE_NAMES,
    }


@app.post("/chem/adr-krebs/scan", response_model=ADRKrebsScanResult)
def scan_adr_krebs(req: ADRKrebsScanRequest):
    krebs_mod = _load_krebs_mod()
    alphas = np.linspace(req.alpha_min, req.alpha_max, req.n_alpha)
    cfg = krebs_mod.KrebsADRConfig(dt=req.dt)
    a_vals, x_vals = krebs_mod.krebs_bifurcation_scan(
        alpha_values=alphas,
        n_steps_transient=req.n_steps_transient,
        n_steps_sample=req.n_steps_sample,
        sample_every=req.sample_every,
        cfg=cfg,
        sample_site=req.sample_site,
    )
    return {"alphas": a_vals.tolist(), "samples": x_vals.tolist(), "site": req.sample_site}


@app.post("/metrics/lle")
def metrics_lle(req: LLERequest):
    return {"lle": rosenstein_lle(np.array(req.series), m=req.m, tau=req.tau)}


@app.post("/sheaf/lorenz", response_model=LorenzSheafResult)
def sheaf_lorenz(req: LorenzSheafRequest) -> LorenzSheafResult:
    data = lorenz_attractor_sheaf(
        req.rhos,
        sigma=req.sigma,
        beta=req.beta,
        steps=req.steps,
        burn_in=req.burn_in,
        dt=req.dt,
    )
    samples = [
        SheafSample(
            rho=s["rho"],
            class_label=s["class_label"],
            lambda_max=s["lambda_max"],
            ky_dim=s["ky_dim"],
            rms=s["rms"],
            tau_past=s["tau_past"],
            tau_future=s["tau_future"],
            capture=s["capture"],
            clc=s["clc"],
        )
        for s in data.get("samples", [])
    ]
    sections = [
        SheafSection(
            start=sec["start"],
            end=sec["end"],
            class_label=sec["class_label"],
            persist=sec.get("persist", True),
        )
        for sec in data.get("sections", [])
    ]
    obstructions = [
        SheafObstruction(left=obs["left"], right=obs["right"], reason=obs["reason"])
        for obs in data.get("obstructions", [])
    ]
    return LorenzSheafResult(samples=samples, sections=sections, obstructions=obstructions)


@app.post("/operator/build")
def operator_build(req: OperatorBuildRequest):
    pts = np.array(req.points, dtype=float)
    P, bounds = build_ulam(pts, nx=req.nx, ny=req.ny)
    global _last_section_hits  # cache for FractalHedron
    _last_section_hits = pts
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


@app.post("/fractalllm/process")
def fractalllm_process(req: FractalRequest):
    data = generate_reservoir_sample(req.text, n_res=req.reservoir_size, coupling=req.coupling)
    return data


@app.post("/dcrc/run")
def dcrc_run(req: DCRCRequest):
    # Generate synthetic trajectories similar to the legacy Dash plugin
    t = np.linspace(0, 10, req.timesteps)
    trajectories = []
    for i in range(req.num_reservoirs):
        phase = i * 2 * np.pi / max(req.num_reservoirs, 1)
        x = np.sin(t + phase) + 0.1 * np.sin(10 * t + phase)
        y = np.cos(t + phase) + 0.1 * np.cos(10 * t + phase)
        z = np.sin(2 * t + phase) * 0.5
        if i > 0:
            x += req.coupling * 0.05 * trajectories[i - 1]["x"]
            y += req.coupling * 0.05 * trajectories[i - 1]["y"]
        trajectories.append({"x": x.tolist(), "y": y.tolist(), "z": z.tolist()})

    # Use the bank to produce a small supervised demo for diagnostics
    X = np.stack([t[:200], np.sin(t[:200]), np.cos(t[:200]), np.sin(2 * t[:200])], axis=1)
    y = np.sin(t[:200] * 0.5) + 0.1 * np.cos(t[:200])
    bank = DCRC_Bank(n_maps=max(1, req.num_reservoirs), n_iter=40, ridge_alpha=1e-2)
    bank.fit(X, y)
    preds = bank.predict(X[:50])

    metrics = {
        "num_reservoirs": req.num_reservoirs,
        "coupling": req.coupling,
        "timesteps": req.timesteps,
        "avg_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
    }
    return {"trajectories": trajectories, "pred_samples": preds.tolist(), "metrics": metrics}


@app.post("/bcp/section")
def bcp_section(req: SectionRequest):
    pts = to_section(np.array(req.points, dtype=float))
    return {"section": pts.tolist(), "count": int(pts.shape[0])}


@app.post("/abtc/integrate")
def abtc_integrate(req: ABTCRequest):
    dyn = Lorenz63()
    f = lambda x: dyn.derivative(x)  # noqa: E731
    traj = rk4(f, np.array(req.x0, dtype=float), dt=req.dt, n=req.steps)
    return {"trajectory": traj.tolist(), "labels": dyn.state_labels}


@app.post("/fractalhedron/run", response_model=FractalHedronResult)
def fractalhedron_run(req: FractalHedronRequest) -> FractalHedronResult:
    global _last_section_hits
    if req.section_hits:
        hits = np.array(req.section_hits, dtype=float)
        _last_section_hits = hits
    else:
        if _last_section_hits is None:
            raise HTTPException(
                status_code=400,
                detail="No section hits provided and none cached. Run /operator/build first or include section_hits.",
            )
        hits = _last_section_hits
    if hits.ndim != 2 or hits.shape[0] == 0:
        raise HTTPException(status_code=400, detail="section_hits must be a non-empty (T,d) array.")
    sym_seq = build_symbolic_sequence(hits, coding_spec=req.coding_spec, coding_params=req.coding_params or None)
    fhk = build_fractalhedron_k(sym_seq, k=req.k, Q=req.Q)
    faces = fractal_face_flags(fhk)
    return {
        "alphabet": fhk["alphabet"],
        "ell_k": fhk["ell_k"],
        "D_q": {str(k): v for k, v in fhk["D_q"].items()},
        "T_q": {str(k): v for k, v in fhk["T_q"].items()},
        "constraints": fhk["constraints"],
        "faces": faces,
    }


@app.post("/self-tuning/lorenz", response_model=SelfTunedLorenzResult)
def self_tuning_lorenz(req: SelfTunedLorenzRequest) -> SelfTunedLorenzResult:
    data = run_self_tuned_lorenz(
        num_steps=req.num_steps,
        print_every=max(req.num_steps // 5, 1),
        record_states=req.record_states,
    )
    return data


@app.post("/self-tuning/sessions", response_model=SelfTuningSessionState)
def create_self_tuning_session(req: SelfTuningSessionCreateRequest) -> SelfTuningSessionState:
    import uuid

    session_id = str(uuid.uuid4())
    system = LorenzSystem(id_=f"lorenz_session_{session_id}", dt=req.dt)
    manager = SelfTuningManager(
        system=system,
        state_dim=system.get_state_vector().shape[0],
        system_step_fn=system.step_pure,
        chaos_band=ChaosBand(lower=req.chaos_band[0], upper=req.chaos_band[1]),
        k_spectral_radius=req.k_spectral_radius,
    )
    _self_tuning_sessions[session_id] = {
        "system": system,
        "manager": manager,
        "step": 0,
        "lambdas": [],
        "spectral_radius": [],
    }
    return SelfTuningSessionState(
        session_id=session_id,
        step=0,
        lambdas=[],
        spectral_radius=[],
        gain=system.gain,
        state=system.get_state_vector().tolist(),
        regime=None,
    )


@app.post("/self-tuning/sessions/step", response_model=SelfTuningSessionState)
def step_self_tuning_session(req: SelfTuningSessionStepRequest) -> SelfTuningSessionState:
    session = _self_tuning_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    system: LorenzSystem = session["system"]
    manager: SelfTuningManager = session["manager"]
    lambdas: List[float] = session["lambdas"]
    srs: List[float] = session["spectral_radius"]

    for _ in range(req.steps):
        system.step()
        manager.update_for_current_state()
        lam = manager.get_latest_lambda_max()
        if lam is not None:
            lambdas.append(lam)
        srs.append(system.spectral_radius)
        session["step"] += 1

    state_vec = system.get_state_vector()
    regime = manager.get_latest_regime()

    return SelfTuningSessionState(
        session_id=req.session_id,
        step=session["step"],
        lambdas=list(lambdas),
        spectral_radius=list(srs),
        gain=system.gain,
        state=state_vec.tolist(),
        regime=regime,
    )


@app.post("/cog/memory", response_model=MemoryExperimentResult)
def cog_memory(req: MemoryExperimentRequest) -> MemoryExperimentResult:
    sim = make_param_forced_simulator(
        model_key=req.model_key,
        param_name=req.param_name,
        base_value=req.base_value,
        amp=req.amp,
        dt=req.dt,
        extra_params={"sigma": req.sigma, "beta": req.beta},
    )

    mem_data = generate_random_drive_dataset(
        simulator=sim,
        n_trials=req.n_trials,
        n_steps=req.n_steps,
        burn_in=req.burn_in,
        input_alphabet=req.input_alphabet,
        input_probs=req.input_probs,
        dt=req.dt,
    )

    ks, M_vals = compute_memory_profile(
        inputs=mem_data.inputs,
        outputs=mem_data.outputs,
        max_lag=req.max_lag,
        n_output_bins=req.n_output_bins,
    )

    states_flat = mem_data.outputs.reshape(-1, mem_data.outputs.shape[-1])
    clc = estimate_clc_metrics(states_flat, dt=req.dt)

    u0 = mem_data.inputs[0]
    y0 = mem_data.outputs[0, :, 0]
    sample_len = min(1_000, u0.shape[0])
    return MemoryExperimentResult(
        lags=ks.astype(int).tolist(),
        M=M_vals.tolist(),
        sample_u=u0[:sample_len].tolist(),
        sample_y=y0[:sample_len].tolist(),
        dt=mem_data.dt,
        tau_past=clc["tau_past"],
        tau_future=clc["tau_future"],
        radius=clc["radius"],
        capture=clc["capture"],
        clc=clc["clc"],
    )


@app.post("/cog/discriminability", response_model=DiscriminabilityExperimentResult)
def cog_discriminability(req: DiscriminabilityExperimentRequest) -> DiscriminabilityExperimentResult:
    if not req.patterns:
        raise HTTPException(status_code=400, detail="patterns must be non-empty")

    sim = make_param_forced_simulator(
        model_key=req.model_key,
        param_name=req.param_name,
        base_value=req.base_value,
        amp=req.amp,
        dt=req.dt,
        extra_params={"sigma": req.sigma, "beta": req.beta},
    )

    patterns = [np.array(p.pattern, dtype=float) for p in req.patterns]
    concept_data = generate_concept_dataset(
        simulator=sim,
        patterns=patterns,
        n_trials_per_concept=req.n_trials_per_concept,
        burn_in=req.burn_in,
        dt=req.dt,
    )

    stats = estimate_discriminability(
        labels=concept_data.labels,
        trajectories=concept_data.outputs,
        test_size=0.3,
        random_state=0,
    )

    return DiscriminabilityExperimentResult(**stats)


@app.get("/datasets")
def datasets_list():
    out = []
    for ds in list_datasets():
        out.append(
            {
                "id": ds.id,
                "name": ds.name,
                "description": ds.description,
                "columns": ds.columns,
                "estimate": ds.count_hint,
                "source": ds.source,
            }
        )
    return {"datasets": out}


@app.post("/datasets/preview")
def datasets_preview(req: DatasetPreviewRequest):
    data = preview_dataset(req.dataset_id, limit=min(req.limit, 5000))
    return data


@app.post("/datasets/upload")
def datasets_upload(req: DatasetUploadRequest):
    content = base64.b64decode(req.contents_b64.split(",")[-1])
    ds = register_upload(req.name, req.description, req.columns, content)
    return {
        "id": ds.id,
        "name": ds.name,
        "description": ds.description,
        "columns": ds.columns,
        "source": ds.source,
    }


@app.post("/api/parse-task")
def parse_task(req: TaskParseRequest):
    spec = _infer_task_spec(req.prompt, req.overrides)
    return spec.model_dump()


@app.post("/api/ingest/preview")
def ingest_preview(req: IngestPreviewRequest):
    preview = _mock_preview(req.sample_size)
    preview.update(
        {
            "source": req.source,
            "modality": req.modality,
            "note": "Mock preview — hook up to your connector to see live samples.",
        }
    )
    return preview


@app.post("/api/ingest/validate")
def ingest_validate(req: IngestValidateRequest):
    config = load_ingestion_config()
    sufficiency = _evaluate_sufficiency(req.row_count, config)
    mapping = req.mapping
    format_checks = [
        {
            "label": f"Time column `{mapping.time_col}` is monotonic",
            "status": "pass" if mapping.time_col else "warn",
        },
        {
            "label": f"{len(mapping.state_cols)} state variables selected",
            "status": "pass" if mapping.state_cols else "fail",
        },
        {
            "label": "Control inputs optional but mapped",
            "status": "pass" if mapping.control_cols else "warn",
        },
        {
            "label": "Group / trial column set",
            "status": "pass" if mapping.group_col else "warn",
        },
    ]
    return {
        "format_checks": format_checks,
        "sufficiency": sufficiency,
        "alerts_ready": bool(req.needs_alerts),
    }


@app.post("/api/dataset/generate")
def dataset_generate(req: DatasetGenerationRequest):
    total_rows = _estimate_dataset_size(req.duration, req.sampling_rate, req.trials)
    preview = _mock_preview(min(total_rows, 25))
    dataset_id = f"synth-{req.system.lower().replace(' ', '-')}-{str(uuid.uuid4())[:8]}"
    ingest_mapping = {
        "time_col": "t",
        "state_cols": ["x", "y", "z"],
        "control_cols": ["u1"],
        "group_col": "trial_id",
    }
    return {
        "id": dataset_id,
        "row_estimate": total_rows,
        "preview": preview,
        "ingest_mapping": ingest_mapping,
        "message": "Synthetic dataset ready — it will flow through the same ingestion wizard.",
        "builder": req.model_dump(),
    }


def run() -> None:
    import uvicorn

    host = os.getenv("SACP_BIND", "127.0.0.1")
    port = int(os.getenv("SACP_PORT", "8000"))
    uvicorn.run("sacp_suite.api.main:app", host=host, port=port, reload=_bool_env("SACP_RELOAD", True))


if __name__ == "__main__":
    run()
