from __future__ import annotations

import math
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import numpy.typing as npt

from sacp_suite.abtc.base import BaseDynamics
from sacp_suite.modules.sacp_x.metrics import rosenstein_lle


def _spectral_entropy(series: np.ndarray) -> float:
    y = np.asarray(series, dtype=float)
    y = y - np.mean(y)
    if y.size == 0:
        return 0.0
    psd = np.abs(np.fft.rfft(y)) ** 2
    psd_sum = float(np.sum(psd)) + 1e-12
    psd = psd / psd_sum
    entropy = -np.sum(psd * np.log(psd + 1e-12))
    norm = np.log(len(psd) + 1e-12)
    if norm <= 0:
        return 0.0
    return float(max(entropy / norm, 0.0))


def _lempel_ziv_complexity(series: np.ndarray) -> float:
    """Simple binary LZ complexity as a proxy for C0."""
    if series.size < 10:
        return 0.0
    thresh = np.median(series)
    bits = (series > thresh).astype(int).tolist()
    seen: set[tuple[int, ...]] = set()
    i = 0
    k = 1
    c = 1
    while True:
        if i + k > len(bits):
            c += 1
            break
        substring = tuple(bits[i : i + k])
        if substring in seen:
            k += 1
        else:
            seen.add(substring)
            c += 1
            i += k
            k = 1
        if i + k > len(bits):
            break
    n = len(bits)
    if n <= 1:
        return 0.0
    return float(c / (n / math.log2(max(n, 2))))


def _zero_one_test(series: np.ndarray) -> float:
    """Lightweight 0-1 chaos proxy."""
    n = len(series)
    if n < 50:
        return 0.0
    c = np.cos(np.arange(n) * math.pi / 5.0)
    s = np.sin(np.arange(n) * math.pi / 5.0)
    p = np.cumsum(series * c)
    q = np.cumsum(series * s)
    msd = p**2 + q**2
    t = np.arange(n)
    if np.allclose(msd, 0):
        return 0.0
    corr = np.corrcoef(t, msd)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(max(min(corr, 1.0), 0.0))


def _complexity_metrics(series: np.ndarray) -> Dict[str, float]:
    se = _spectral_entropy(series)
    c0 = _lempel_ziv_complexity(series)
    return {"SE": se, "C0": c0}


def simulate_fde(
    dynamics: BaseDynamics,
    params: Dict[str, float],
    x0: npt.NDArray[np.float64],
    *,
    t_max: float = 500.0,
    dt: float = 0.01,
    t_transient: float = 0.0,
    sample_stride: int = 1,
    compute_0_1: bool = False,
    compute_complexity: bool = False,
    observable: Literal["x", "y", "z"] = "x",
    return_currents: bool = False,
) -> Tuple[Dict[str, np.ndarray], float | None, Dict[str, object]]:
    params_full = dynamics.default_params()
    params_full.update(params or {})

    x = np.array(x0, dtype=float).copy()
    n_steps = max(int(t_max / dt), 1)
    burn_steps = max(int(t_transient / dt), 0)
    sample_stride = max(int(sample_stride), 1)

    q_vec = np.asarray(dynamics.orders(params_full), dtype=float)
    q = float(np.mean(q_vec))
    q = min(max(q, 0.01), 1.0)
    scale = dt**q / float(math.gamma(q + 1.0))

    traj_t: list[float] = [0.0]
    traj_x: list[float] = [float(x[0])]
    traj_y: list[float] = [float(x[1]) if x.shape[0] > 1 else 0.0]
    traj_z: list[float] = [float(x[2]) if x.shape[0] > 2 else 0.0]
    curr_JS: list[np.ndarray] = []
    curr_JC: list[np.ndarray] = []
    curr_JR: list[np.ndarray] = []
    curr_JE: list[np.ndarray] = []

    t = 0.0
    diverged = False
    for step in range(n_steps):
        deriv = dynamics.rhs(t, x, params_full)
        x = x + scale * deriv
        t += dt

        diverged = not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6

        if step >= burn_steps and (step - burn_steps) % sample_stride == 0:
            traj_t.append(t)
            traj_x.append(float(x[0]))
            traj_y.append(float(x[1]))
            traj_z.append(float(x[2]) if x.shape[0] > 2 else 0.0)

            if return_currents:
                decomp = dynamics.rhs_decomposed(t, x, params_full)
                curr_JS.append(np.asarray(decomp.get("JS"), dtype=float))
                curr_JC.append(np.asarray(decomp.get("JC"), dtype=float))
                curr_JR.append(np.asarray(decomp.get("JR"), dtype=float))
                curr_JE.append(np.asarray(decomp.get("JE"), dtype=float))

        if diverged:
            break

    traj = {
        "t": np.nan_to_num(np.asarray(traj_t, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "x": np.nan_to_num(np.asarray(traj_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "y": np.nan_to_num(np.asarray(traj_y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        "z": np.nan_to_num(np.asarray(traj_z, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
    }

    obs_idx = {"x": 0, "y": 1, "z": 2}.get(observable, 0)
    obs_series = traj[["x", "y", "z"][obs_idx]]

    lle = None
    if obs_series.size > 50 and np.all(np.isfinite(obs_series)):
        lle = rosenstein_lle(obs_series)

    extras: Dict[str, object] = {}
    extras["diverged"] = diverged
    if compute_0_1:
        extras["k01"] = _zero_one_test(obs_series)
    if compute_complexity:
        extras["complexity"] = _complexity_metrics(obs_series)
    if return_currents and curr_JS:
        extras["currents"] = {
            "JS": np.stack(curr_JS),
            "JC": np.stack(curr_JC),
            "JR": np.stack(curr_JR),
            "JE": np.stack(curr_JE),
        }

    return traj, lle, extras


def bifurcation_scan(
    dynamics: BaseDynamics,
    params: Dict[str, float],
    *,
    scan_param: Literal["q", "k1"],
    start: float,
    stop: float,
    num: int,
    observable: Literal["x", "y", "z"] = "x",
    t_max: float = 500.0,
    dt: float = 0.01,
    t_transient: float = 100.0,
    sample_stride: int = 5,
) -> Dict[str, object]:
    values = np.linspace(start, stop, num)
    samples: list[np.ndarray] = []
    for val in values:
        p_local = params.copy()
        p_local[scan_param] = float(val)
        traj, _, _ = simulate_fde(
            dynamics,
            p_local,
            dynamics.default_state(),
            t_max=t_max,
            dt=dt,
            t_transient=t_transient,
            sample_stride=sample_stride,
            compute_0_1=False,
            compute_complexity=False,
            observable=observable,
            return_currents=False,
        )
        obs_series = traj[observable]
        tail = obs_series[-200:] if obs_series.size > 200 else obs_series
        samples.append(np.asarray(tail, dtype=float))

    return {
        "param_values": values.tolist(),
        "samples": samples,
    }


def compute_complexity_grid(
    dynamics: BaseDynamics,
    base_params: Dict[str, float],
    *,
    q_min: float,
    q_max: float,
    q_steps: int,
    k1_min: float,
    k1_max: float,
    k1_steps: int,
    observable: Literal["x", "y", "z"] = "x",
) -> Dict[str, object]:
    q_values = np.linspace(q_min, q_max, q_steps)
    k1_values = np.linspace(k1_min, k1_max, k1_steps)
    se_grid = np.zeros((len(k1_values), len(q_values)))
    c0_grid = np.zeros_like(se_grid)

    for i_k, k1 in enumerate(k1_values):
        for i_q, q in enumerate(q_values):
            params = {**base_params, "k1": float(k1), "q": float(q)}
            traj, _, extras = simulate_fde(
                dynamics,
                params,
                dynamics.default_state(),
                t_max=30.0,
                dt=0.004,
                t_transient=10.0,
                sample_stride=6,
                compute_0_1=False,
                compute_complexity=True,
                observable=observable,
                return_currents=False,
            )
            comp = extras.get("complexity") if isinstance(extras, dict) else None
            if isinstance(comp, dict):
                se_grid[i_k, i_q] = float(comp.get("SE", 0.0))
                c0_grid[i_k, i_q] = float(comp.get("C0", 0.0))

    return {
        "q_values": q_values.tolist(),
        "k1_values": k1_values.tolist(),
        "se": se_grid,
        "c0": c0_grid,
    }


def basin_scan(
    dynamics: BaseDynamics,
    params: Dict[str, float],
    *,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    nx: int,
    nz: int,
    y0: float,
    t_max: float,
    dt: float,
    t_transient: float,
) -> Dict[str, object]:
    max_cells = 40000
    if nx * nz > max_cells:
        shrink = math.sqrt(max_cells / float(nx * nz))
        nx = max(5, int(nx * shrink))
        nz = max(5, int(nz * shrink))

    x_values = np.linspace(x_min, x_max, nx)
    z_values = np.linspace(z_min, z_max, nz)
    labels = np.zeros((len(z_values), len(x_values)), dtype=int)

    for iz, z0 in enumerate(z_values):
        for ix, x0_val in enumerate(x_values):
            x0 = np.array([x0_val, y0, z0], dtype=float)
            traj, lle, _ = simulate_fde(
                dynamics,
                params,
                x0,
                t_max=t_max,
                dt=dt,
                t_transient=t_transient,
                sample_stride=max(1, int(1.0 / max(dt, 1e-3))),
                compute_0_1=False,
                compute_complexity=False,
                observable="x",
                return_currents=False,
            )
            if not traj["x"].size:
                labels[iz, ix] = 2
                continue
            final_state = np.array([traj["x"][-1], traj["y"][-1], traj["z"][-1]], dtype=float)
            if not np.all(np.isfinite(final_state)) or np.linalg.norm(final_state) > 1e4:
                labels[iz, ix] = 2  # divergent
            elif lle is not None and lle > 0.1:
                labels[iz, ix] = 1  # chaotic / hidden attractor
            else:
                labels[iz, ix] = 0  # point / regular

    return {
        "x_values": x_values.tolist(),
        "z_values": z_values.tolist(),
        "labels": labels,
    }
