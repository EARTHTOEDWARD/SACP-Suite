"""
Bouquet torus stack utilities for tuning toward the bouquet cone and scanning alpha.

This is a light, dependency-free scaffold:
- Bouquet geometry (tau(d) quadratic fit from the RAST note).
- ADR ring dynamics per layer (InstrumentedADR-like).
- CLC proxy metrics.
- Self-tuning gate to stay inside the bouquet speed bound while targeting a CLC band.
- Convenience runners for single tuning runs and alpha sweeps.

All outputs are plain Python types so they can be JSON-serialised by FastAPI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ---------- Geometry ----------


def tau_quadratic(d: float) -> float:
    """Quadratic latency fit from RAST: tau(d) ~ 5e-6 d^2 + 3.5e-5 seconds."""
    return 5.0e-6 * (d ** 2) + 3.5e-5


@dataclass
class TorusLayerGeom:
    index: int
    L: float       # circumference
    d: float       # code distance
    tau: float     # latency tau(d)
    c: float       # emergent classical speed c_i = L / tau


@dataclass
class BouquetGeom:
    layers: List[TorusLayerGeom]
    alpha: float

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def build_bouquet_geom(
    n_layers: int,
    alpha: float,
    L0: float = 10.0,
    d0: int = 3,
) -> BouquetGeom:
    layers = []
    L = L0
    d = d0
    for i in range(n_layers):
        tau = tau_quadratic(d)
        c = L / tau
        layers.append(TorusLayerGeom(index=i, L=L, d=d, tau=tau, c=c))
        L *= alpha
        d += 2
    return BouquetGeom(layers=layers, alpha=alpha)


# ---------- ADR core ----------


@dataclass
class ADRLayerConfig:
    N: int = 16
    dt: float = 0.01
    a: float = 1.0
    gamma: float = 0.2
    mu: float = 0.1
    sigma: float = 0.5
    eta: float = 0.5
    k_init: float = 0.4
    seed: int = 0


class InstrumentedADR:
    """
    Minimal ADR integrator with resource loop and ring coupling.
    """

    def __init__(
        self,
        N: int,
        dt: float,
        a: float,
        gamma: float,
        mu: float,
        sigma: float,
        eta: float,
        F: float = 0.0,
        omega: float = 1.0,
        k_init: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.N = N
        self.dt = dt
        self.a = a
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.F = F
        self.omega = omega
        self.k = k_init

        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal(N) * 0.1
        self.v = rng.standard_normal(N) * 0.1
        self.r = rng.standard_normal(N) * 0.1
        self.t = 0.0

    def step(self) -> None:
        lap = np.roll(self.x, 1) + np.roll(self.x, -1) - 2.0 * self.x
        drive = self.eta * np.roll(self.r, 1) + self.F * math.cos(self.omega * self.t)
        dv = (
            -self.gamma * self.v
            + self.a * self.x
            - np.power(self.x, 3)
            + self.k * lap
            + drive
        )
        dx = self.v
        dr = -self.mu * self.r + self.sigma * np.power(self.x, 2)

        self.v = self.v + self.dt * dv
        self.x = self.x + self.dt * dx
        self.r = self.r + self.dt * dr
        self.t += self.dt


class ADRLayer:
    """Wraps an ADR integrator with bouquet latency gating for vertical coupling."""

    def __init__(self, geom: TorusLayerGeom, cfg: ADRLayerConfig, layer_seed: int = 0) -> None:
        self.geom = geom
        self.cfg = cfg
        self.adr = InstrumentedADR(
            N=cfg.N,
            dt=cfg.dt,
            a=cfg.a,
            gamma=cfg.gamma,
            mu=cfg.mu,
            sigma=cfg.sigma,
            eta=cfg.eta,
            F=0.0,
            omega=1.0,
            k_init=cfg.k_init,
            seed=layer_seed,
        )
        self.latency_steps = max(1, int(math.ceil(geom.tau / cfg.dt)))
        self.time_steps = 0

    def step(self, vertical_field: Optional[np.ndarray] = None) -> None:
        self.adr.step()
        self.time_steps += 1
        if vertical_field is not None and self.time_steps % self.latency_steps == 0:
            self.adr.v = self.adr.v + vertical_field


# ---------- Bouquet stack ----------


class ADRBouquetStack:
    def __init__(
        self,
        bouquet_geom: BouquetGeom,
        layer_cfg: ADRLayerConfig,
        k_vert: float = 0.0,
        seed: int = 123,
    ) -> None:
        self.geom = bouquet_geom
        self.k_vert = k_vert
        self.layers: List[ADRLayer] = []
        rng = np.random.default_rng(seed)
        for layer_geom in bouquet_geom.layers:
            lay_seed = int(rng.integers(0, 2**31 - 1))
            self.layers.append(ADRLayer(layer_geom, layer_cfg, layer_seed=lay_seed))

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def N(self) -> int:
        return self.layers[0].cfg.N

    def _vertical_fields(self) -> List[Optional[np.ndarray]]:
        if self.k_vert == 0.0 or self.n_layers < 2:
            return [None for _ in self.layers]
        xs = [lay.adr.x.copy() for lay in self.layers]
        fields = []
        for i in range(self.n_layers):
            above = xs[(i + 1) % self.n_layers]
            below = xs[(i - 1) % self.n_layers]
            here = xs[i]
            vert_lap = above + below - 2.0 * here
            fields.append(self.k_vert * vert_lap.reshape(self.N))
        return fields

    def step(self) -> None:
        v_fields = self._vertical_fields()
        for lay, vf in zip(self.layers, v_fields):
            lay.step(vertical_field=vf)

    def run(self, n_steps: int = 5000, log_every: int = 10) -> Dict[str, Any]:
        logs = {
            "time": [],
            "x_layers": [],
        }
        for step in range(n_steps):
            self.step()
            if step % log_every == 0:
                logs["time"].append(self.layers[0].adr.t)
                x_snapshot = np.stack([lay.adr.x.copy() for lay in self.layers], axis=0)
                logs["x_layers"].append(x_snapshot)
        logs["time"] = np.array(logs["time"])
        logs["x_layers"] = np.stack(logs["x_layers"], axis=0)  # (T, n_layers, N)
        return logs


# ---------- CLC proxy ----------


def _autocorr_first_crossing(x: np.ndarray, max_lag: int) -> int:
    x = x - x.mean()
    if x.std() == 0:
        return 1
    ac_full = np.correlate(x, x, mode="full")
    ac = ac_full[len(ac_full) // 2 :]
    ac = ac / ac[0]
    target = 1.0 / math.e
    for lag in range(1, min(max_lag, len(ac))):
        if ac[lag] < target:
            return lag
    return min(max_lag, len(ac) - 1)


def compute_clc_proxy(
    log: Dict[str, np.ndarray],
    dt: float,
    ac_max_lag: int = 200,
) -> tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Lightweight CLC proxy:
    - tau_past, tau_future: first autocorr crossing of 1/e (seconds).
    - R_spatial: correlation-based spatial reach (site units).
    """
    x = log["x"]  # (T, N)
    T, N = x.shape
    tau_past = np.zeros(N)
    tau_future = np.zeros(N)
    R_spatial = np.zeros(N)

    for j in range(N):
        tau_past[j] = _autocorr_first_crossing(x[:, j], ac_max_lag) * dt
        tau_future[j] = tau_past[j]  # symmetric proxy

    # Spatial reach: count correlated neighbors (|corr| > 0.2) along ring.
    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = x_centered.T @ x_centered / max(T - 1, 1)
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var) + 1e-12)
    corr = cov / denom
    for j in range(N):
        mask = np.abs(corr[j]) > 0.2
        reach = np.sum(mask) - 1  # exclude self
        R_spatial[j] = max(1, reach / 2.0)  # approximate radius in sites

    C_inf = 1.0 / (1.0 + np.exp(-np.abs(x).mean(axis=0)))
    S_node = C_inf * np.sqrt(tau_past * tau_future) * R_spatial
    S_ring = float(np.mean(S_node))

    diag = {
        "C_inf": C_inf,
        "tau_past": tau_past,
        "tau_future": tau_future,
        "R_spatial": R_spatial,
    }
    return S_node, S_ring, diag


def analyze_clc_vs_bouquet(
    stack: ADRBouquetStack,
    logs: Dict[str, Any],
    dt: float,
) -> Dict[str, Any]:
    T, n_layers, N = logs["x_layers"].shape
    results = []
    for i, layer in enumerate(stack.layers):
        log_i = {
            "x": logs["x_layers"][:, i, :],
            "R_local": np.zeros((T, N)),
        }
        _, S_ring, diag = compute_clc_proxy(
            log_i,
            dt=dt,
            ac_max_lag=min(200, T - 1 if T > 1 else 1),
        )
        C_inf = diag["C_inf"]
        tau_past = diag["tau_past"]
        tau_future = diag["tau_future"]
        R_spatial = diag["R_spatial"]

        tau_clc = math.sqrt(
            max(tau_past.mean(), 1e-8) * max(tau_future.mean(), 1e-8)
        )
        R_sites = R_spatial.mean()
        L = layer.geom.L
        tau_geom = layer.geom.tau
        c_geom = layer.geom.c

        delta_ell = L / float(N)
        v_cog = (R_sites * delta_ell) / max(tau_clc, 1e-8)

        bouquet_ok = v_cog <= c_geom

        results.append(
            {
                "layer_index": i,
                "L": L,
                "d": layer.geom.d,
                "tau_geom": tau_geom,
                "c_geom": c_geom,
                "S_ring": S_ring,
                "C_inf_mean": float(C_inf.mean()),
                "tau_clc": tau_clc,
                "R_sites": R_sites,
                "v_cog": v_cog,
                "bouquet_ok": bouquet_ok,
                "bound_R_sites_max": N * tau_clc / max(tau_geom, 1e-8),
            }
        )

    return {"per_layer": results}


# ---------- Self-tuning gate ----------


@dataclass
class SelfTuningConfig:
    target_S_ring: float = 1.5
    gain_eta: float = 0.05
    gain_k: float = 0.02
    eta_bounds: tuple[float, float] = (0.05, 1.5)
    k_bounds: tuple[float, float] = (0.05, 1.5)
    v_margin: float = 0.05  # allow v_cog up to (1+v_margin) * c_geom before penalizing


class SelfTuningGate:
    """Adjust eta and k per layer to target a CLC band while respecting bouquet speed."""

    def __init__(self, cfg: SelfTuningConfig) -> None:
        self.cfg = cfg

    def _clip(self, val: float, bounds: tuple[float, float]) -> float:
        return min(bounds[1], max(bounds[0], val))

    def tune_layer(
        self,
        layer: ADRLayer,
        window_x: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        log_i = {"x": window_x, "R_local": np.zeros_like(window_x)}
        _, S_ring, diag = compute_clc_proxy(
            log_i, dt=dt, ac_max_lag=min(200, len(window_x) - 1)
        )

        tau_clc = math.sqrt(
            max(diag["tau_past"].mean(), 1e-8) * max(diag["tau_future"].mean(), 1e-8)
        )
        R_sites = diag["R_spatial"].mean()
        delta_ell = layer.geom.L / float(layer.cfg.N)
        v_cog = (R_sites * delta_ell) / max(tau_clc, 1e-8)

        speed_ratio = v_cog / max(layer.geom.c, 1e-12)
        over_speed = max(0.0, speed_ratio - (1.0 + self.cfg.v_margin))

        eta_new = layer.adr.eta
        k_new = layer.adr.k

        if over_speed > 0:
            eta_new -= self.cfg.gain_eta * over_speed
            k_new -= self.cfg.gain_k * over_speed
        else:
            err = self.cfg.target_S_ring - S_ring
            eta_new += self.cfg.gain_eta * err
            k_new += self.cfg.gain_k * err

        layer.adr.eta = self._clip(eta_new, self.cfg.eta_bounds)
        layer.adr.k = self._clip(k_new, self.cfg.k_bounds)

        return {
            "S_ring": S_ring,
            "v_cog": v_cog,
            "c_geom": layer.geom.c,
            "eta": layer.adr.eta,
            "k": layer.adr.k,
            "speed_ratio": speed_ratio,
        }


def run_with_controller(
    stack: ADRBouquetStack,
    n_steps: int,
    log_every: int,
    control_every: int,
    control_window: int,
    tuner: SelfTuningGate,
) -> Dict[str, Any]:
    logs = {"time": [], "x_layers": []}
    tuning_log: List[Dict[str, Any]] = []
    window: List[np.ndarray] = []

    for step in range(n_steps):
        stack.step()

        if step % log_every == 0:
            logs["time"].append(stack.layers[0].adr.t)
            x_snapshot = np.stack([lay.adr.x.copy() for lay in stack.layers], axis=0)
            logs["x_layers"].append(x_snapshot)
            window.append(x_snapshot)
            if len(window) > control_window:
                window.pop(0)

        if (step > 0) and (step % control_every == 0) and len(window) >= control_window:
            window_arr = np.stack(window, axis=0)  # (W, n_layers, N)
            tuning_step = []
            for i, lay in enumerate(stack.layers):
                diag = tuner.tune_layer(
                    lay,
                    window_arr[:, i, :],
                    dt=stack.layers[0].cfg.dt,
                )
                diag["layer_index"] = i
                tuning_step.append(diag)
            tuning_log.append({"step": step, "tuning": tuning_step})

    logs["time"] = np.array(logs["time"])
    logs["x_layers"] = np.stack(logs["x_layers"], axis=0)
    return {"logs": logs, "tuning_log": tuning_log}


# ---------- Convenience runners ----------


def _summarize_layers(per_layer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in per_layer:
        out.append(
            {
                "layer_index": int(r["layer_index"]),
                "c_geom": float(r["c_geom"]),
                "v_cog": float(r["v_cog"]),
                "bouquet_ok": bool(r["bouquet_ok"]),
                "S_ring": float(r["S_ring"]),
                "tau_clc": float(r["tau_clc"]),
            }
        )
    return out


def run_bouquet(
    alpha: float,
    n_layers: int = 4,
    N: int = 12,
    dt: float = 0.01,
    n_steps: int = 2000,
    log_every: int = 10,
    k_vert: float = 0.05,
    seed: int = 123,
) -> Dict[str, Any]:
    geom = build_bouquet_geom(n_layers=n_layers, alpha=alpha)
    cfg = ADRLayerConfig(N=N, dt=dt)
    stack = ADRBouquetStack(geom, cfg, k_vert=k_vert, seed=seed)
    logs = stack.run(n_steps=n_steps, log_every=log_every)
    rep = analyze_clc_vs_bouquet(stack, logs, dt=dt)
    return {
        "alpha": alpha,
        "monotone_pass": all(
            geom.layers[i].c >= geom.layers[i + 1].c for i in range(len(geom.layers) - 1)
        ),
        "per_layer": rep["per_layer"],
        "summary": _summarize_layers(rep["per_layer"]),
    }


def run_bouquet_tuning(
    alpha: float,
    n_layers: int = 4,
    N: int = 12,
    dt: float = 0.01,
    n_steps: int = 3000,
    log_every: int = 10,
    k_vert: float = 0.05,
    seed: int = 123,
    control_every: int = 200,
    control_window: int = 40,
    tuning: Optional[SelfTuningConfig] = None,
) -> Dict[str, Any]:
    geom = build_bouquet_geom(n_layers=n_layers, alpha=alpha)
    cfg = ADRLayerConfig(N=N, dt=dt)
    stack = ADRBouquetStack(geom, cfg, k_vert=k_vert, seed=seed)
    tuner = SelfTuningGate(tuning or SelfTuningConfig())

    out = run_with_controller(
        stack=stack,
        n_steps=n_steps,
        log_every=log_every,
        control_every=control_every,
        control_window=control_window,
        tuner=tuner,
    )
    rep = analyze_clc_vs_bouquet(stack, out["logs"], dt=dt)
    return {
        "alpha": alpha,
        "monotone_pass": all(
            geom.layers[i].c >= geom.layers[i + 1].c for i in range(len(geom.layers) - 1)
        ),
        "per_layer": rep["per_layer"],
        "summary": _summarize_layers(rep["per_layer"]),
        "tuning_log": out["tuning_log"],
    }


def run_alpha_sweep(
    alphas: List[float],
    n_layers: int = 4,
    N: int = 10,
    dt: float = 0.01,
    n_steps: int = 1500,
    log_every: int = 10,
    k_vert: float = 0.05,
    seed: int = 21,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for alpha in alphas:
        geom = build_bouquet_geom(n_layers=n_layers, alpha=alpha)
        cfg = ADRLayerConfig(N=N, dt=dt)
        stack = ADRBouquetStack(geom, cfg, k_vert=k_vert, seed=seed)
        logs = stack.run(n_steps=n_steps, log_every=log_every)
        rep = analyze_clc_vs_bouquet(stack, logs, dt=dt)
        results.append(
            {
                "alpha": alpha,
                "monotone_pass": all(
                    geom.layers[i].c >= geom.layers[i + 1].c for i in range(len(geom.layers) - 1)
                ),
                "per_layer": _summarize_layers(rep["per_layer"]),
            }
        )
    return results
