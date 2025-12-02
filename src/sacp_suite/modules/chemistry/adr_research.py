"""Research-grade ADR ring plugin and experiments (CLC, MI, FTLE, synergy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import math
import numpy as np
from scipy.integrate import solve_ivp

from sacp_suite.core.plugin_api import BaseDynamics, registry


# ---------------------------------------------------------------------------
# Core ADR ring model (Duffing sites + resource loop)
# ---------------------------------------------------------------------------


@dataclass
class ADRParams:
    N: int
    a: np.ndarray
    gamma: np.ndarray
    kappa: float
    eta: float
    mu: float
    sigma: float


def adr_rhs(
    t: float,
    state: np.ndarray,
    params: ADRParams,
    probe_fn: Callable[[float], float],
) -> np.ndarray:
    """
    Minimal ADR ring ODE:
      ẍ_i + γ_i ẋ_i + ∂V_i = κ Δx_i + η r_{i-1} + u_i(t)
      ṙ_i = -μ r_i + σ x_i^2
    with V_i(x) = 1/4 x^4 - a_i/2 x^2 (double-well).
    state = [x (N), v (N), r (N)].
    """
    N = params.N
    x = state[:N]
    v = state[N : 2 * N]
    r = state[2 * N :]

    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    lap = x_ip1 + x_im1 - 2.0 * x

    u = np.zeros_like(x)
    u[0] = probe_fn(t)

    dx = v
    dv = (
        -params.gamma * v
        + params.a * x
        - x**3
        + params.kappa * lap
        + params.eta * np.roll(r, 1)
        + u
    )
    dr = -params.mu * r + params.sigma * x**2
    return np.concatenate([dx, dv, dr])


def simulate_adr_ring(
    params: ADRParams,
    T: float,
    dt: float,
    probe_fn: Callable[[float], float],
    x0: np.ndarray | None = None,
    burn_in: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate ADR ring and return (t, probe, x_traj) after optional burn-in."""
    N = params.N
    n_steps = int(T / dt)
    t_eval = np.linspace(0, T, n_steps)

    if x0 is None:
        rng = np.random.default_rng()
        x_init = 0.1 * rng.standard_normal(N)
        v_init = 0.1 * rng.standard_normal(N)
        r_init = np.zeros(N)
        state0 = np.concatenate([x_init, v_init, r_init])
    else:
        if x0.shape[0] != 3 * N:
            raise ValueError(f"x0 must have length 3N={3*N}")
        state0 = x0

    sol = solve_ivp(
        lambda t, y: adr_rhs(t, y, params, probe_fn),
        (0, T),
        state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    t = sol.t
    Y = sol.y.T
    x_traj = Y[:, :N]
    probe = np.array([probe_fn(tt) for tt in t])

    if burn_in > 0:
        mask = t >= burn_in
        t = t[mask]
        probe = probe[mask]
        x_traj = x_traj[mask]

    return t, probe, x_traj


# ---------------------------------------------------------------------------
# CLC + information metrics
# ---------------------------------------------------------------------------


def autocorrelation_time(x: np.ndarray, dt: float, c_thresh: float = 0.1) -> float:
    """Estimate correlation time from first threshold crossing of autocorr."""
    x = x - x.mean()
    ac_full = np.correlate(x, x, mode="full")
    ac = ac_full[ac_full.size // 2 :]
    if ac[0] != 0:
        ac = ac / ac[0]
    for k, val in enumerate(ac):
        if k == 0:
            continue
        if val < c_thresh:
            return k * dt
    return (len(ac) - 1) * dt


def spatial_radius(x_traj: np.ndarray, i: int, c_thresh: float = 0.2) -> float:
    """Mean ring distance to sites correlated with site i."""
    _, N = x_traj.shape
    X = x_traj - x_traj.mean(axis=0, keepdims=True)
    C = np.corrcoef(X, rowvar=False)
    corr_row = C[i]
    indices = [j for j in range(N) if abs(corr_row[j]) > c_thresh and j != i]
    if not indices:
        return 0.0
    dists = [min(abs(j - i), N - abs(j - i)) for j in indices]
    return float(np.mean(dists))


def local_coherence(x_traj: np.ndarray, i: int, window: int = 50, thresh: float = 0.7) -> float:
    """Fraction of windows where site i is coherent with its neighbours."""
    T, N = x_traj.shape
    x = x_traj[:, i]
    x_ip1 = x_traj[:, (i + 1) % N]
    x_im1 = x_traj[:, (i - 1) % N]
    neigh = 0.5 * (x_ip1 + x_im1)

    count = 0
    total = 0
    for start in range(0, T - window, window):
        seg_x = x[start : start + window]
        seg_n = neigh[start : start + window]
        seg_x = seg_x - seg_x.mean()
        seg_n = seg_n - seg_n.mean()
        denom = np.linalg.norm(seg_x) * np.linalg.norm(seg_n)
        if denom == 0:
            continue
        c = float(np.dot(seg_x, seg_n) / denom)
        total += 1
        if c > thresh:
            count += 1
    if total == 0:
        return 0.0
    return count / total


def compute_clc_scores(x_traj: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute Cognitive Light Cone proxy S_CLC,i:
      S_i ≈ C_inf,i * sqrt(τ_past,i * τ_future,i) * R_i
    Here τ_future ≈ τ_past (symmetric proxy).
    """
    _, N = x_traj.shape
    S = np.zeros(N)
    for i in range(N):
        tau = autocorrelation_time(x_traj[:, i], dt)
        R = spatial_radius(x_traj, i)
        C_inf = local_coherence(x_traj, i)
        S[i] = C_inf * math.sqrt(tau * tau) * R
    return S


def mutual_information_hist(x: np.ndarray, y: np.ndarray, n_bins: int = 32) -> float:
    """Histogram-based MI estimator in nats."""
    x_edges = np.histogram_bin_edges(x, bins=n_bins)
    y_edges = np.histogram_bin_edges(y, bins=n_bins)
    joint_hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    joint_prob = joint_hist / np.sum(joint_hist)
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = joint_prob > 0
    mi = np.sum(joint_prob[nz] * np.log(joint_prob[nz] / px_py[nz]))
    return float(mi)


def transfer_entropy_discrete(
    source: np.ndarray, target: np.ndarray, lag: int = 1, n_bins: int = 16
) -> float:
    """Discrete TE(source -> target) with one-step lag (histogram proxy)."""
    assert source.shape == target.shape
    x = source
    y = target
    T = len(x)
    if T <= lag + 1:
        return 0.0

    s_past = x[:-lag]
    y_past = y[:-lag]
    y_now = y[lag:]

    def _digitize(z: np.ndarray) -> np.ndarray:
        hist, edges = np.histogram(z, bins=n_bins)
        del hist  # unused; edges needed
        return np.clip(np.digitize(z, edges[:-1]) - 1, 0, n_bins - 1)

    s_b = _digitize(s_past)
    yp_b = _digitize(y_past)
    y_b = _digitize(y_now)

    joint = np.zeros((n_bins, n_bins, n_bins), dtype=float)
    joint_y = np.zeros((n_bins, n_bins), dtype=float)
    joint_yp = np.zeros((n_bins,), dtype=float)

    for si, ypi, yi in zip(s_b, yp_b, y_b):
        joint[si, ypi, yi] += 1.0
        joint_y[ypi, yi] += 1.0
        joint_yp[ypi] += 1.0

    joint /= np.sum(joint)
    joint_y /= np.sum(joint_y)
    joint_yp /= np.sum(joint_yp)

    eps = 1e-12
    te = 0.0
    for si in range(n_bins):
        for ypi in range(n_bins):
            denom = np.sum(joint[si, ypi, :]) + eps
            denom2 = np.sum(joint_y[ypi, :]) + eps
            for yi in range(n_bins):
                p_xyz = joint[si, ypi, yi]
                if p_xyz <= 0:
                    continue
                p_y_given_syp = p_xyz / denom
                p_y_given_yp = (joint_y[ypi, yi] + eps) / denom2
                te += p_xyz * np.log(p_y_given_syp / p_y_given_yp)
    return float(te)


def estimate_ftle(
    state0: np.ndarray,
    params: ADRParams,
    probe_fn: Callable[[float], float],
    T: float,
    dt: float,
    eps: float = 1e-6,
) -> float:
    """Finite-time Lyapunov exponent via twin-trajectory RK4 method."""
    N_state = state0.size
    n_steps = int(T / dt)
    delta = np.random.randn(N_state)
    delta /= np.linalg.norm(delta)
    delta *= eps
    s = state0.copy()
    s_pert = state0 + delta
    t = 0.0
    sum_log = 0.0

    def rhs(time: float, y: np.ndarray) -> np.ndarray:
        return adr_rhs(time, y, params, probe_fn)

    for _ in range(n_steps):
        k1 = rhs(t, s)
        k2 = rhs(t + 0.5 * dt, s + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, s + 0.5 * dt * k2)
        k4 = rhs(t + dt, s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        k1p = rhs(t, s_pert)
        k2p = rhs(t + 0.5 * dt, s_pert + 0.5 * dt * k1p)
        k3p = rhs(t + 0.5 * dt, s_pert + 0.5 * dt * k2p)
        k4p = rhs(t + dt, s_pert + dt * k3p)
        s_pert = s_pert + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)

        d = s_pert - s
        norm = np.linalg.norm(d)
        if norm == 0:
            t += dt
            continue
        sum_log += np.log(norm / eps)
        d = (eps / norm) * d
        s_pert = s + d
        t += dt

    return sum_log / (n_steps * dt)


def toy_efe(mi: np.ndarray, x_traj: np.ndarray, alpha: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Toy Expected Free Energy proxy per site: G_i = alpha Var[x_i] - gamma MI(probe, x_i)."""
    var = np.var(x_traj, axis=0)
    return alpha * var - gamma * mi


def run_isolated_vs_ring_demo() -> Dict[str, Any]:
    """
    Quick empirical test: isolated sites vs coupled ADR ring.
    Returns dict with metrics for downstream use instead of printing only.
    """
    N = 12
    dt = 0.02
    T = 200.0
    burn_in = 50.0
    f0 = 0.1
    amp = 0.2

    def probe_fn(t: float) -> float:
        return amp * math.sin(2 * math.pi * f0 * t)

    a = 1.0 * np.ones(N)
    gamma = 0.2 * np.ones(N)
    mu = 1.0
    sigma = 0.5

    params_iso = ADRParams(N=N, a=a, gamma=gamma, kappa=0.0, eta=0.0, mu=mu, sigma=sigma)
    t_iso, probe_iso, x_iso = simulate_adr_ring(params_iso, T, dt, probe_fn, burn_in=burn_in)
    S_iso = compute_clc_scores(x_iso, dt)
    mi_iso = np.array([mutual_information_hist(probe_iso, x_iso[:, i]) for i in range(N)])

    params_ring = ADRParams(N=N, a=a, gamma=gamma, kappa=0.3, eta=0.4, mu=mu, sigma=sigma)
    t_ring, probe_ring, x_ring = simulate_adr_ring(params_ring, T, dt, probe_fn, burn_in=burn_in)
    S_ring = compute_clc_scores(x_ring, dt)
    mi_ring = np.array([mutual_information_hist(probe_ring, x_ring[:, i]) for i in range(N)])
    mi_ring_mean = mutual_information_hist(probe_ring, x_ring.mean(axis=1))

    G_iso = toy_efe(mi_iso, x_iso)
    G_ring = toy_efe(mi_ring, x_ring)

    return {
        "t_iso": t_iso,
        "t_ring": t_ring,
        "probe_iso": probe_iso,
        "probe_ring": probe_ring,
        "S_iso": S_iso,
        "S_ring": S_ring,
        "mi_iso": mi_iso,
        "mi_ring": mi_ring,
        "mi_ring_mean": mi_ring_mean,
        "G_iso": G_iso,
        "G_ring": G_ring,
    }


# ---------------------------------------------------------------------------
# Adaptive controller (learn kappa, eta online to lower toy EFE)
# ---------------------------------------------------------------------------


def simulate_adaptive_adr_ring(
    params: ADRParams,
    T_total: float,
    dt: float,
    probe_fn: Callable[[float], float],
    window: float = 10.0,
    kappa_bounds: Tuple[float, float] = (0.0, 1.0),
    eta_bounds: Tuple[float, float] = (0.0, 1.0),
    step_kappa: float = 0.02,
    step_eta: float = 0.02,
    alpha_risk: float = 1.0,
    gamma_epi: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-loop ADR simulation where (kappa, eta) adapt via a sign-gradient rule
    to reduce G = alpha_risk * Var[x] - gamma_epi * MI(probe, ring-mean).
    """
    N = params.N
    n_steps_window = max(2, int(window / dt))
    n_windows = int(T_total / window)

    rng = np.random.default_rng()
    x_init = 0.1 * rng.standard_normal(N)
    v_init = 0.1 * rng.standard_normal(N)
    r_init = np.zeros(N)
    state = np.concatenate([x_init, v_init, r_init])

    t_all: list[np.ndarray] = []
    probe_all: list[np.ndarray] = []
    x_all: list[np.ndarray] = []
    kappa_hist: list[float] = []
    eta_hist: list[float] = []
    G_hist: list[float] = []

    dir_kappa = 1.0
    dir_eta = 1.0
    prev_G: float | None = None
    t0 = 0.0

    for _ in range(n_windows):
        t_start = t0
        t_end = t0 + window
        t_eval = np.linspace(t_start, t_end, n_steps_window)

        sol = solve_ivp(
            lambda t, y: adr_rhs(t, y, params, probe_fn),
            (t_start, t_end),
            state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )
        t_w = sol.t
        Y_w = sol.y.T
        x_w = Y_w[:, :N]
        probe_w = np.array([probe_fn(tt) for tt in t_w])

        t_all.append(t_w)
        probe_all.append(probe_w)
        x_all.append(x_w)

        ring_mean = x_w.mean(axis=1)
        mi_ring = mutual_information_hist(probe_w, ring_mean)
        var_ring = float(np.var(x_w))
        G = alpha_risk * var_ring - gamma_epi * mi_ring

        kappa_hist.append(params.kappa)
        eta_hist.append(params.eta)
        G_hist.append(G)

        if prev_G is not None and G >= prev_G:
            dir_kappa *= -1.0
            dir_eta *= -1.0

        params.kappa = float(np.clip(params.kappa + dir_kappa * step_kappa, kappa_bounds[0], kappa_bounds[1]))
        params.eta = float(np.clip(params.eta + dir_eta * step_eta, eta_bounds[0], eta_bounds[1]))

        prev_G = G
        state = sol.y[:, -1]
        t0 = t_end

    t_concat = np.concatenate(t_all)
    probe_concat = np.concatenate(probe_all)
    x_concat = np.vstack(x_all)

    return t_concat, probe_concat, x_concat, np.array(kappa_hist), np.array(eta_hist), np.array(G_hist)


def run_adaptive_controller_demo() -> Dict[str, Any]:
    """Run adaptive ADR demo and return metrics for inspection."""
    N = 12
    dt = 0.02
    T_total = 200.0
    f0 = 0.1
    amp = 0.2

    def probe_fn(t: float) -> float:
        return amp * math.sin(2 * math.pi * f0 * t)

    params = ADRParams(
        N=N,
        a=1.0 * np.ones(N),
        gamma=0.2 * np.ones(N),
        kappa=0.01,
        eta=0.01,
        mu=1.0,
        sigma=0.5,
    )

    t, probe, x, k_hist, e_hist, G_hist = simulate_adaptive_adr_ring(
        params,
        T_total=T_total,
        dt=dt,
        probe_fn=probe_fn,
        window=10.0,
        kappa_bounds=(0.0, 0.6),
        eta_bounds=(0.0, 0.8),
        step_kappa=0.02,
        step_eta=0.02,
        alpha_risk=1.0,
        gamma_epi=1.0,
    )

    half_idx = len(t) // 2
    x_final = x[half_idx:]
    probe_final = probe[half_idx:]

    S_final = compute_clc_scores(x_final, dt)
    mi_sites_final = np.array([mutual_information_hist(probe_final, x_final[:, i]) for i in range(N)])
    mi_ring_final = mutual_information_hist(probe_final, x_final.mean(axis=1))

    return {
        "t": t,
        "probe": probe,
        "x": x,
        "kappa_hist": k_hist,
        "eta_hist": e_hist,
        "G_hist": G_hist,
        "mean_clc": float(S_final.mean()),
        "mean_mi_sites": float(mi_sites_final.mean()),
        "mi_ring": float(mi_ring_final),
    }


# ---------------------------------------------------------------------------
# Two-ring cooperation experiment (energy + information synergy)
# ---------------------------------------------------------------------------


def two_ring_rhs(
    t: float,
    state: np.ndarray,
    params: ADRParams,
    probe_fn: Callable[[float], float],
    k_inter: float,
) -> np.ndarray:
    """Two coupled ADR rings; ring 1 is probed directly, ring 2 only via coupling."""
    N = params.N
    x1 = state[0:N]
    v1 = state[N : 2 * N]
    r1 = state[2 * N : 3 * N]

    x2 = state[3 * N : 4 * N]
    v2 = state[4 * N : 5 * N]
    r2 = state[5 * N : 6 * N]

    ip1 = np.roll(np.arange(N), -1)
    im1 = np.roll(np.arange(N), 1)

    u1 = np.zeros(N, dtype=float)
    u1[0] = probe_fn(t)
    u2 = np.zeros(N, dtype=float)

    dVdx1 = x1**3 - params.a * x1
    dVdx2 = x2**3 - params.a * x2

    lap1 = params.kappa * (x1[ip1] + x1[im1] - 2.0 * x1)
    lap2 = params.kappa * (x2[ip1] + x2[im1] - 2.0 * x2)

    cat1 = params.eta * r1[im1]
    cat2 = params.eta * r2[im1]

    y1_mean = float(np.mean(x1))
    y2_mean = float(np.mean(x2))
    c1 = k_inter * (y2_mean - y1_mean)
    c2 = -c1

    dx1 = v1
    dv1 = -params.gamma * v1 - dVdx1 + lap1 + cat1 + u1 + c1
    dr1 = -params.mu * r1 + params.sigma * x1**2

    dx2 = v2
    dv2 = -params.gamma * v2 - dVdx2 + lap2 + cat2 + u2 + c2
    dr2 = -params.mu * r2 + params.sigma * x2**2

    return np.concatenate([dx1, dv1, dr1, dx2, dv2, dr2])


def simulate_two_rings(
    params: ADRParams,
    T: float,
    dt: float,
    probe_fn: Callable[[float], float],
    k_inter: float = 0.0,
    burn_in: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate two ADR rings with optional inter-ring coupling."""
    rng = np.random.default_rng(seed)
    N = params.N
    n_steps = int(T / dt)
    t_eval = np.linspace(0.0, T, n_steps)

    x1_0 = 0.1 * rng.standard_normal(size=N)
    v1_0 = 0.1 * rng.standard_normal(size=N)
    r1_0 = np.zeros(N)
    x2_0 = 0.1 * rng.standard_normal(size=N)
    v2_0 = 0.1 * rng.standard_normal(size=N)
    r2_0 = np.zeros(N)

    state0 = np.concatenate([x1_0, v1_0, r1_0, x2_0, v2_0, r2_0])

    sol = solve_ivp(
        lambda t, y: two_ring_rhs(t, y, params, probe_fn, k_inter),
        (0.0, T),
        state0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    t = sol.t
    Y = sol.y.T
    x1 = Y[:, 0:N]
    v1 = Y[:, N : 2 * N]
    x2 = Y[:, 3 * N : 4 * N]
    v2 = Y[:, 4 * N : 5 * N]
    probe = np.array([probe_fn(tt) for tt in t])

    if burn_in > 0.0:
        mask = t >= burn_in
        t = t[mask]
        x1 = x1[mask]
        x2 = x2[mask]
        v1 = v1[mask]
        v2 = v2[mask]
        probe = probe[mask]

    v_both = np.stack([v1, v2], axis=1)
    return t, probe, x1, x2, v_both


def power_series_from_velocities(v_both: np.ndarray, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous power proxy P = sum_i gamma_i v_i^2 per ring."""
    v1 = v_both[:, 0, :]
    v2 = v_both[:, 1, :]
    P1 = np.sum(gamma * v1**2, axis=1)
    P2 = np.sum(gamma * v2**2, axis=1)
    return P1, P2


def energy_baseline_isolated(P1: np.ndarray, P2: np.ndarray, dt: float) -> float:
    """External energy if rings are isolated (no buffer): integral of total power."""
    return dt * float(np.sum(P1 + P2))


def energy_with_shared_buffer(
    P1: np.ndarray,
    P2: np.ndarray,
    dt: float,
    alpha_dep: float = 0.2,
    alpha_wdr: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    External energy with a shared stigmergic buffer R(t).
      - total demand D = P1 + P2
      - withdraw up to alpha_wdr * R (not exceeding D)
      - remaining demand is paid externally; fraction alpha_dep deposited into R
    """
    assert P1.shape == P2.shape
    T_steps = P1.shape[0]
    R = 0.0
    R_series = np.zeros(T_steps, dtype=float)
    E_coupled = 0.0

    for k in range(T_steps):
        D = P1[k] + P2[k]
        withdraw = min(D, alpha_wdr * R)
        Q_ext = D - withdraw
        deposit = alpha_dep * Q_ext
        R = R + (deposit - withdraw) * dt
        if R < 0.0:
            R = 0.0
        E_coupled += Q_ext * dt
        R_series[k] = R

    return E_coupled, R_series


def run_two_ring_synergy_experiment() -> Dict[str, Any]:
    """
    Two-ring ADR cooperation test with energy + information synergy indices.
    Returns a metrics dict so callers can log/plot without parsing stdout.
    """
    N = 16
    dt = 0.01
    T = 200.0
    burn_in = 50.0
    f0 = 0.15
    amp = 0.2

    def probe_fn(t: float) -> float:
        return amp * math.sin(2.0 * math.pi * f0 * t)

    params = ADRParams(
        N=N,
        a=1.0 * np.ones(N),
        gamma=0.2 * np.ones(N),
        kappa=0.25,
        eta=0.4,
        mu=1.0,
        sigma=0.5,
    )

    t_iso, probe_iso, x1_iso, x2_iso, v_iso = simulate_two_rings(
        params, T=T, dt=dt, probe_fn=probe_fn, k_inter=0.0, burn_in=burn_in, seed=1
    )
    dt_eff = t_iso[1] - t_iso[0]
    P1_iso, P2_iso = power_series_from_velocities(v_iso, params.gamma)
    E_iso = energy_baseline_isolated(P1_iso, P2_iso, dt_eff)
    ring1_iso = np.mean(x1_iso, axis=1)
    ring2_iso = np.mean(x2_iso, axis=1)
    I1_iso = mutual_information_hist(probe_iso, ring1_iso)
    I2_iso = mutual_information_hist(probe_iso, ring2_iso)

    k_inter = 0.2
    t_c, probe_c, x1_c, x2_c, v_c = simulate_two_rings(
        params, T=T, dt=dt, probe_fn=probe_fn, k_inter=k_inter, burn_in=burn_in, seed=2
    )
    dt_eff_c = t_c[1] - t_c[0]
    P1_c, P2_c = power_series_from_velocities(v_c, params.gamma)
    E_coupled_ext, R_series = energy_with_shared_buffer(P1_c, P2_c, dt_eff_c, alpha_dep=0.2, alpha_wdr=0.5)
    S_E = (E_iso - E_coupled_ext) / E_iso

    ring1_c = np.mean(x1_c, axis=1)
    ring2_c = np.mean(x2_c, axis=1)
    I1_c = mutual_information_hist(probe_c, ring1_c)
    I2_c = mutual_information_hist(probe_c, ring2_c)
    I_total_iso = I1_iso + I2_iso
    I_total_c = I1_c + I2_c
    S_info = (I_total_c - I_total_iso) / (I_total_iso + 1e-12)

    return {
        "E_iso": E_iso,
        "E_coupled_ext": E_coupled_ext,
        "energy_synergy": S_E,
        "I1_iso": I1_iso,
        "I2_iso": I2_iso,
        "I1_c": I1_c,
        "I2_c": I2_c,
        "info_synergy": S_info,
        "R_final": float(R_series[-1]),
        "k_inter": k_inter,
        "t_iso": t_iso,
        "t_c": t_c,
    }


# ---------------------------------------------------------------------------
# Plugin wrapper so ADR ring is available via the registry
# ---------------------------------------------------------------------------


def _to_array(val: float | Sequence[float], n: int) -> np.ndarray:
    arr = np.asarray(val, dtype=float)
    if arr.size == 1:
        return np.full(n, float(arr), dtype=float)
    if arr.size != n:
        raise ValueError(f"Expected length {n}, got {arr.size}")
    return arr


def _params_from_dict(d: Dict[str, Any]) -> ADRParams:
    N = int(d.get("N", 12))
    a = _to_array(d.get("a", 1.0), N)
    gamma = _to_array(d.get("gamma", 0.2), N)
    kappa = float(d.get("kappa", 0.3))
    eta = float(d.get("eta", 0.4))
    mu = float(d.get("mu", 1.0))
    sigma = float(d.get("sigma", 0.5))
    return ADRParams(N=N, a=a, gamma=gamma, kappa=kappa, eta=eta, mu=mu, sigma=sigma)


class ADRRingDynamics(BaseDynamics):
    """Generic ADR ring plugin using the research-grade ADR model."""

    def get_default_params(self) -> Dict[str, float]:
        return {"N": 12, "a": 1.0, "gamma": 0.2, "kappa": 0.3, "eta": 0.4, "mu": 1.0, "sigma": 0.5}

    def default_state(self) -> np.ndarray:
        p = _params_from_dict(self.params)
        rng = np.random.default_rng(42)
        x = 0.1 * rng.standard_normal(p.N)
        v = 0.1 * rng.standard_normal(p.N)
        r = np.zeros(p.N, dtype=float)
        return np.concatenate([x, v, r])

    def derivative(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:  # noqa: ARG002
        params = _params_from_dict(self.params)
        return adr_rhs(t, state, params, lambda _: 0.0)

    @property
    def name(self) -> str:
        return "Autocatalytic Duffing Ring"

    @property
    def state_labels(self) -> list[str]:
        p = _params_from_dict(self.params)
        labels = [f"x_{i}" for i in range(p.N)] + [f"v_{i}" for i in range(p.N)] + [f"r_{i}" for i in range(p.N)]
        return labels


registry.register_dynamics("adr_ring", ADRRingDynamics)

