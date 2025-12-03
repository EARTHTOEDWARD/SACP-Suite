"""ADR–Bioelectric lattice prototype with self-tuning and experiment hooks.

Implements:
  - 7-tile 1D lattice, each tile has a 4-site ADR ring (Vmem, channels, pumps, morph program)
  - Fast ADR dynamics + slow self-tuning of biases, damping, and ring drive
  - Experiment hooks: negative island + flip-back, wound + repair
Dependencies: numpy, scipy.integrate.solve_ivp
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

# Lattice dimensions
N_TILES = 7  # tiles indexed by j
N_SITES = 4  # sites per tile: 0 Vmem, 1 channels, 2 pumps, 3 morph program


@dataclass
class ADRBioelectricParams:
    # Local double-well coefficients per site
    a: np.ndarray = np.array([1.0, 0.9, 0.8, 0.7], dtype=float)

    # Base damping per site (site 0 uses gamma1[j], others use base)
    gamma_base: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    # Ring coupling inside each tile
    k_ring: float = 0.2

    # Spatial coupling between tiles for Vmem (site 0)
    K_space: float = 0.15

    # Resource dynamics
    eta0: float = 0.6  # base ring drive, scaled by eta[j]
    mu: float = 0.1
    sigma: float = 0.4

    # Self-tuning learning rates (slow)
    eps_V: float = 1e-3
    eps_M: float = 1e-3
    eps_gamma: float = 1e-4
    eps_eta: float = 1e-4

    # Target patterns (allow scalar or per-tile array; normalised in __post_init__)
    V_star: np.ndarray | float | None = None
    M_star: np.ndarray | float | None = None

    # Weights in slow-cost
    w_V: float = 1.0
    w_M: float = 0.3
    w_sync: float = 0.5
    w_lambda: float = 0.0  # set >0 when Lyapunov gating is plugged in

    # Chaos band (for Lyapunov gating, currently placeholder)
    lambda_min: float = 0.0
    lambda_max: float = 0.5

    # Bounds / clipping for slow parameters
    gamma_min: float = 0.05
    gamma_max: float = 0.6
    eta_min: float = 0.1
    eta_max: float = 1.5
    h_bound: float = 5.0  # |h1|, |h4| <= h_bound

    def __post_init__(self) -> None:
        base_V = float(np.sqrt(self.a[0]))
        base_M = float(np.sqrt(self.a[3]))

        # V_star normalisation
        if self.V_star is None:
            self.V_star = np.full(N_TILES, base_V, dtype=float)
        else:
            V = np.asarray(self.V_star, dtype=float)
            if V.ndim == 0:
                self.V_star = np.full(N_TILES, float(V), dtype=float)
            elif V.shape == (N_TILES,):
                self.V_star = V
            else:
                raise ValueError(f"V_star must be scalar or shape ({N_TILES},)")

        # M_star normalisation
        if self.M_star is None:
            self.M_star = np.full(N_TILES, base_M, dtype=float)
        else:
            M = np.asarray(self.M_star, dtype=float)
            if M.ndim == 0:
                self.M_star = np.full(N_TILES, float(M), dtype=float)
            elif M.shape == (N_TILES,):
                self.M_star = M
            else:
                raise ValueError(f"M_star must be scalar or shape ({N_TILES},)")


# ---------- State pack/unpack ----------


def unpack_state(y: np.ndarray):
    """Unpack flat state vector into components."""
    idx = 0
    n = N_SITES * N_TILES
    x = y[idx : idx + n].reshape(N_SITES, N_TILES)
    idx += n
    v = y[idx : idx + n].reshape(N_SITES, N_TILES)
    idx += n
    r = y[idx : idx + n].reshape(N_SITES, N_TILES)
    idx += n
    h1 = y[idx : idx + N_TILES]
    idx += N_TILES
    h4 = y[idx : idx + N_TILES]
    idx += N_TILES
    gamma1 = y[idx : idx + N_TILES]
    idx += N_TILES
    eta = y[idx : idx + N_TILES]
    return x, v, r, h1, h4, gamma1, eta


def pack_state(x: np.ndarray, v: np.ndarray, r: np.ndarray, h1: np.ndarray, h4: np.ndarray, gamma1: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Pack components into a flat state vector."""
    return np.concatenate([x.reshape(-1), v.reshape(-1), r.reshape(-1), h1, h4, gamma1, eta])


def make_head_tail_pattern(params: ADRBioelectricParams, head_scale: float = 1.1, tail_scale: float = 0.7) -> ADRBioelectricParams:
    """Encode a head–tail gradient into V_star / M_star."""
    base_V = float(np.sqrt(params.a[0]))
    base_M = float(np.sqrt(params.a[3]))
    V_star_vec = base_V * np.linspace(head_scale, tail_scale, N_TILES)
    M_star_vec = base_M * np.linspace(head_scale, tail_scale, N_TILES)
    params.V_star = V_star_vec
    params.M_star = M_star_vec
    params.__post_init__()
    return params


def make_bipolar_pattern(params: ADRBioelectricParams) -> ADRBioelectricParams:
    """Encode a simple bipolar (center-dipped) pattern."""
    base_V = float(np.sqrt(params.a[0]))
    base_M = float(np.sqrt(params.a[3]))
    V_star_vec = np.array([1.0, 1.0, 0.5, 0.2, 0.5, 1.0, 1.0]) * base_V
    M_star_vec = np.array([1.0, 1.0, 0.5, 0.2, 0.5, 1.0, 1.0]) * base_M
    params.V_star = V_star_vec
    params.M_star = M_star_vec
    params.__post_init__()
    return params


# Default non-uniform pattern helper: head–tail gradient (tile 0 → 6)
def apply_default_head_tail(params: ADRBioelectricParams, head_scale: float = 1.1, tail_scale: float = 0.8) -> ADRBioelectricParams:
    """Apply a head–tail gradient to V_star / M_star and re-normalize."""
    base_V = float(np.sqrt(params.a[0]))
    base_M = float(np.sqrt(params.a[3]))
    V_star_vec = base_V * np.linspace(head_scale, tail_scale, N_TILES)
    M_star_vec = base_M * np.linspace(head_scale, tail_scale, N_TILES)
    params.V_star = V_star_vec
    params.M_star = M_star_vec
    params.__post_init__()
    return params


def make_initial_state(params: ADRBioelectricParams, mode: str = "healthy") -> np.ndarray:
    """
    mode:
      - "healthy": all tiles near healthy attractor
      - "healthy_with_negative_center": center tile near negative attractor
      - "random_wound_center": center tiles scrambled (wound)
    """
    a = params.a
    xH = np.array([+np.sqrt(a[0]), +np.sqrt(a[1]), +np.sqrt(a[2]), +np.sqrt(a[3])])
    xN = -xH

    x = np.zeros((N_SITES, N_TILES))
    v = np.zeros_like(x)
    r = np.zeros_like(x)

    for j in range(N_TILES):
        x[:, j] = xH + 0.05 * np.random.randn(N_SITES)
        r[:, j] = params.sigma * a / params.mu

    if mode == "healthy_with_negative_center":
        j0 = N_TILES // 2
        x[:, j0] = xN + 0.05 * np.random.randn(N_SITES)

    if mode == "random_wound_center":
        for j in [2, 3, 4]:
            x[:, j] = 0.5 * np.random.randn(N_SITES)
            r[:, j] = 0.0

    h1 = np.zeros(N_TILES)
    h4 = np.zeros(N_TILES)
    gamma1 = np.full(N_TILES, params.gamma_base[0])
    eta = np.ones(N_TILES)
    return pack_state(x, v, r, h1, h4, gamma1, eta)


# ---------- Fast dynamics ----------


def neighbor_indices_1d(j: int) -> list[int]:
    """Neighbor indices for tile j in open 1D line."""
    nbrs: list[int] = []
    if j > 0:
        nbrs.append(j - 1)
    if j < N_TILES - 1:
        nbrs.append(j + 1)
    return nbrs


def rhs_fast(
    t: float,
    x: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    h1: np.ndarray,
    h4: np.ndarray,
    gamma1: np.ndarray,
    eta: np.ndarray,
    params: ADRBioelectricParams,
    control_fn,
):
    """Fast ADR layer (x, v, r)."""
    a = params.a
    gamma_base = params.gamma_base
    k_ring = params.k_ring
    K_space = params.K_space
    eta0 = params.eta0
    mu = params.mu
    sigma = params.sigma

    u = control_fn(t)

    dx = np.zeros_like(x)
    dv = np.zeros_like(v)
    dr = np.zeros_like(r)

    for j in range(N_TILES):
        for i in range(N_SITES):
            ip = (i + 1) % N_SITES
            im = (i - 1) % N_SITES
            gamma_ij = gamma1[j] if i == 0 else gamma_base[i]
            Vprime = x[i, j] ** 3 - a[i] * x[i, j]
            lap_ring = x[ip, j] + x[im, j] - 2.0 * x[i, j]
            drive_res = eta0 * eta[j] * r[im, j]
            dx[i, j] = v[i, j]
            dv[i, j] = -gamma_ij * v[i, j] - Vprime + k_ring * lap_ring + drive_res + u[i, j]
            dr[i, j] = -mu * r[i, j] + sigma * x[i, j] ** 2

        # spatial Laplacian on Vmem (site 0)
        i = 0
        nbrs = neighbor_indices_1d(j)
        if nbrs:
            lap_space = -len(nbrs) * x[i, j]
            for k in nbrs:
                lap_space += x[i, k]
            dv[i, j] += K_space * lap_space

        # homeostatic biases
        dv[0, j] += h1[j]
        dv[3, j] += h4[j]

    return dx, dv, dr


# ---------- Slow self-tuning ----------


def lyapunov_estimates_placeholder(x: np.ndarray, v: np.ndarray, r: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Stub for per-tile Lyapunov estimates (extend later)."""
    return np.zeros(N_TILES)


def rhs_slow(
    x: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    h1: np.ndarray,
    h4: np.ndarray,
    gamma1: np.ndarray,
    eta: np.ndarray,
    params: ADRBioelectricParams,
):
    """Slow self-tuning dynamics for biases and couplings."""
    V = x[0, :]
    M = x[3, :]
    V_star = params.V_star
    M_star = params.M_star
    w_V = params.w_V
    w_M = params.w_M
    w_sync = params.w_sync
    w_lambda = params.w_lambda
    eps_V = params.eps_V
    eps_M = params.eps_M
    eps_gamma = params.eps_gamma
    eps_eta = params.eps_eta
    lambda_min = params.lambda_min
    lambda_max = params.lambda_max

    V_bar = np.zeros_like(V)
    for j in range(N_TILES):
        nbrs = neighbor_indices_1d(j)
        V_bar[j] = np.mean([V[k] for k in nbrs]) if nbrs else V[j]

    lambdas = lyapunov_estimates_placeholder(x, v, r)
    dU_dlambda = np.zeros(N_TILES)
    if w_lambda != 0.0:
        for j in range(N_TILES):
            lam = lambdas[j]
            if lam < lambda_min:
                dU_dlambda[j] = 2.0 * (lam - lambda_min)
            elif lam > lambda_max:
                dU_dlambda[j] = 2.0 * (lam - lambda_max)
        dU_dlambda *= w_lambda

    dh1 = np.zeros_like(h1)
    dh4 = np.zeros_like(h4)
    dgamma1 = np.zeros_like(gamma1)
    deta = np.zeros_like(eta)

    for j in range(N_TILES):
        dG_dV = 2.0 * w_V * (V[j] - V_star[j]) + 2.0 * w_sync * (V[j] - V_bar[j])
        dh1[j] = -eps_V * dG_dV

        dG_dM = 2.0 * w_M * (M[j] - M_star[j])
        dh4[j] = -eps_M * dG_dM

        if w_lambda != 0.0:
            dU_dl = dU_dlambda[j]
            dgamma1[j] = +eps_gamma * dU_dl
            deta[j] = -eps_eta * dU_dl

    return dh1, dh4, dgamma1, deta


def rhs_full(t: float, y: np.ndarray, params: ADRBioelectricParams, control_fn):
    """Combined fast + slow RHS for solve_ivp."""
    x, v, r, h1, h4, gamma1, eta = unpack_state(y)
    dx, dv, dr = rhs_fast(t, x, v, r, h1, h4, gamma1, eta, params, control_fn)
    dh1, dh4, dgamma1, deta = rhs_slow(x, v, r, h1, h4, gamma1, eta, params)
    return pack_state(dx, dv, dr, dh1, dh4, dgamma1, deta)


# ---------- Lyapunov estimation helpers ----------


def pack_fast(x: np.ndarray, v: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(-1), v.reshape(-1), r.reshape(-1)])


def unpack_fast(y_flat: np.ndarray):
    n = N_SITES * N_TILES
    x = y_flat[0:n].reshape(N_SITES, N_TILES)
    v = y_flat[n : 2 * n].reshape(N_SITES, N_TILES)
    r = y_flat[2 * n : 3 * n].reshape(N_SITES, N_TILES)
    return x, v, r


def make_lyap_rhs(params: ADRBioelectricParams, h1: np.ndarray, h4: np.ndarray, gamma1: np.ndarray, eta: np.ndarray, control_fn):
    """Build RHS for real + shadow trajectories for Lyapunov estimation."""

    def lyap_rhs(t: float, z: np.ndarray) -> np.ndarray:
        n_fast = 3 * N_SITES * N_TILES
        y_real = z[0:n_fast]
        y_shadow = z[n_fast:]
        x, v, r = unpack_fast(y_real)
        xs, vs, rs = unpack_fast(y_shadow)
        dx, dv, dr = rhs_fast(t, x, v, r, h1, h4, gamma1, eta, params, control_fn)
        dxs, dvs, drs = rhs_fast(t, xs, vs, rs, h1, h4, gamma1, eta, params, control_fn)
        dy_real = pack_fast(dx, dv, dr)
        dy_shadow = pack_fast(dxs, dvs, drs)
        return np.concatenate([dy_real, dy_shadow])

    return lyap_rhs


def estimate_lyapunov_per_tile(
    x0: np.ndarray,
    v0: np.ndarray,
    r0: np.ndarray,
    h1: np.ndarray,
    h4: np.ndarray,
    gamma1: np.ndarray,
    eta: np.ndarray,
    params: ADRBioelectricParams,
    control_fn,
    T_total: float = 200.0,
    T_chunk: float = 2.0,
    delta0: float = 1e-6,
    max_step: float = 0.05,
):
    """
    Finite-time Lyapunov estimate per tile using twin trajectories.
    Returns lambdas shape (N_TILES,).
    """
    y_real0 = pack_fast(x0, v0, r0)

    eps = 1e-8
    xs0 = x0 + eps * np.random.randn(*x0.shape)
    vs0 = v0 + eps * np.random.randn(*v0.shape)
    rs0 = r0 + eps * np.random.randn(*r0.shape)

    def renormalise_tilewise(x, v, r, xs, vs, rs):
        for j in range(N_TILES):
            dx = xs[:, j] - x[:, j]
            dv = vs[:, j] - v[:, j]
            dr = rs[:, j] - r[:, j]
            d2 = float(np.sum(dx * dx + dv * dv + dr * dr))
            if d2 == 0.0:
                dx = 1e-9 * np.random.randn(N_SITES)
                dv = 1e-9 * np.random.randn(N_SITES)
                dr = 1e-9 * np.random.randn(N_SITES)
                d2 = float(np.sum(dx * dx + dv * dv + dr * dr))
            scale = delta0 / np.sqrt(d2)
            xs[:, j] = x[:, j] + scale * dx
            vs[:, j] = v[:, j] + scale * dv
            rs[:, j] = r[:, j] + scale * dr
        return xs, vs, rs

    xs0, vs0, rs0 = renormalise_tilewise(x0, v0, r0, xs0, vs0, rs0)
    y_shadow0 = pack_fast(xs0, vs0, rs0)
    z = np.concatenate([y_real0, y_shadow0])

    lyap_rhs = make_lyap_rhs(params, h1, h4, gamma1, eta, control_fn)

    n_chunks = int(T_total / T_chunk)
    log_sum = np.zeros(N_TILES, dtype=float)
    t0 = 0.0

    for _ in range(n_chunks):
        t1 = t0 + T_chunk
        sol = solve_ivp(
            lyap_rhs,
            (t0, t1),
            z,
            method="RK45",
            max_step=max_step,
            rtol=1e-6,
            atol=1e-9,
            dense_output=False,
        )
        z = sol.y[:, -1]
        n_fast = 3 * N_SITES * N_TILES
        y_real = z[0:n_fast]
        y_shadow = z[n_fast:]
        x, v, r = unpack_fast(y_real)
        xs, vs, rs = unpack_fast(y_shadow)

        for j in range(N_TILES):
            dx = xs[:, j] - x[:, j]
            dv = vs[:, j] - v[:, j]
            dr = rs[:, j] - r[:, j]
            d = np.sqrt(np.sum(dx * dx + dv * dv + dr * dr))
            if d > 0:
                log_sum[j] += np.log(d / delta0)

        xs, vs, rs = renormalise_tilewise(x, v, r, xs, vs, rs)
        y_real = pack_fast(x, v, r)
        y_shadow = pack_fast(xs, vs, rs)
        z = np.concatenate([y_real, y_shadow])
        t0 = t1

    lambdas = log_sum / max(T_total, 1e-9)
    return lambdas


# ---------- Control + experiments ----------


def make_control_negative_island(
    t_kick_start: float = 200.0,
    t_kick_end: float = 210.0,
    F_kick: float = 2.5,
    t_flip_start: float = 400.0,
    t_flip_end: float = 430.0,
    F_flip: float = 2.5,
    enable_flip: bool = True,
):
    """Control: push center tile negative, optionally flip back later."""
    j0 = N_TILES // 2

    def control_fn(t: float) -> np.ndarray:
        u = np.zeros((N_SITES, N_TILES))
        if t_kick_start <= t <= t_kick_end:
            u[0, j0] += -F_kick
            u[1, j0] += -0.7 * F_kick
            u[3, j0] += -0.5 * F_kick
        if enable_flip and (t_flip_start <= t <= t_flip_end):
            u[0, j0] += +F_flip
            u[1, j0] += +1.5
            u[3, j0] += +1.0
        return u

    return control_fn


def integrate_with_clipping(
    y0: np.ndarray,
    t_span: tuple[float, float],
    params: ADRBioelectricParams,
    control_fn,
    max_step: float = 0.1,
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """Integrate with solve_ivp and clip slow params at recorded steps."""

    def wrapped_rhs(t: float, y: np.ndarray) -> np.ndarray:
        return rhs_full(t, y, params, control_fn)

    sol = solve_ivp(
        wrapped_rhs,
        t_span,
        y0,
        method="RK45",
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=False,
        vectorized=False,
    )

    Y = sol.y.T.copy()
    for k in range(Y.shape[0]):
        x, v, r, h1, h4, gamma1, eta = unpack_state(Y[k])
        h1 = np.clip(h1, -params.h_bound, +params.h_bound)
        h4 = np.clip(h4, -params.h_bound, +params.h_bound)
        gamma1 = np.clip(gamma1, params.gamma_min, params.gamma_max)
        eta = np.clip(eta, params.eta_min, params.eta_max)
        Y[k] = pack_state(x, v, r, h1, h4, gamma1, eta)
    sol.y = Y.T
    return sol


def run_experiment_negative_island(params: ADRBioelectricParams | None = None, self_tuning: bool = True, enable_flip: bool = True):
    """Negative island experiment with optional flip and self-tuning."""
    if params is None:
        params = ADRBioelectricParams()
    if not self_tuning:
        params = ADRBioelectricParams(
            **{
                **params.__dict__,
                "eps_V": 0.0,
                "eps_M": 0.0,
                "eps_gamma": 0.0,
                "eps_eta": 0.0,
            }
        )

    y0 = make_initial_state(params, mode="healthy")
    control_fn = make_control_negative_island(enable_flip=enable_flip)
    sol = integrate_with_clipping(y0, (0.0, 500.0), params, control_fn)

    t = sol.t
    X = sol.y
    n = N_SITES * N_TILES
    x_traj = X[0:n, :].reshape(N_SITES, N_TILES, -1)
    V_traj = x_traj[0]  # Vmem per tile
    return t, V_traj, sol


def run_experiment_wound(params: ADRBioelectricParams | None = None, t_wound: float = 200.0, t_final: float = 500.0):
    """Wound + repair experiment: scramble center tiles then self-tune back."""
    if params is None:
        params = ADRBioelectricParams()

    # If caller passed in a params with scalar targets, ensure arrays are set up
    params.__post_init__()

    y0 = make_initial_state(params, mode="healthy")
    control_zero = lambda t: np.zeros((N_SITES, N_TILES))  # noqa: E731
    sol1 = integrate_with_clipping(y0, (0.0, t_wound), params, control_zero)

    y_wound = sol1.y[:, -1].copy()
    x, v, r, h1, h4, gamma1, eta = unpack_state(y_wound)
    for j in [2, 3, 4]:
        x[:, j] = 0.5 * np.random.randn(N_SITES)
        v[:, j] = 0.0
        r[:, j] = 0.0
    y_wound = pack_state(x, v, r, h1, h4, gamma1, eta)

    params_repair = ADRBioelectricParams(**params.__dict__)
    params_repair.K_space = 0.25
    sol2 = integrate_with_clipping(y_wound, (t_wound, t_final), params_repair, control_zero)

    t = np.concatenate([sol1.t, sol2.t])
    Y = np.concatenate([sol1.y, sol2.y], axis=1)
    n = N_SITES * N_TILES
    x_traj = Y[0:n, :].reshape(N_SITES, N_TILES, -1)
    V_traj = x_traj[0]
    return t, V_traj, Y


# ---------- Convenience extraction + comparison helpers ----------


def extract_site_trajectory(Y: np.ndarray, site_idx: int) -> np.ndarray:
    """Return trajectory for a given site index (shape: N_TILES x T) from packed Y."""
    n = N_SITES * N_TILES
    x_traj = Y[0:n, :].reshape(N_SITES, N_TILES, -1)
    return x_traj[site_idx]


def run_wound_head_tail_comparison(
    head_scale: float = 1.1,
    tail_scale: float = 0.8,
    t_wound: float = 200.0,
    t_final: float = 500.0,
    lyap_probe: bool = False,
):
    """
    Run wound experiment with a head–tail pattern:
      - no self-tuning (all eps_* = 0)
      - with self-tuning (default eps_*)
    Optionally estimate per-tile Lyapunov exponents on final states.
    Returns a dict with trajectories and optional lambdas.
    """
    # Patterned params with self-tuning
    params = apply_default_head_tail(ADRBioelectricParams(), head_scale=head_scale, tail_scale=tail_scale)

    # No-tune copy
    params_no_tune = ADRBioelectricParams(
        **{
            **params.__dict__,
            "eps_V": 0.0,
            "eps_M": 0.0,
            "eps_gamma": 0.0,
            "eps_eta": 0.0,
        }
    )
    params_no_tune = apply_default_head_tail(params_no_tune, head_scale=head_scale, tail_scale=tail_scale)

    # Run wound scenarios
    t_nt, V_nt, Y_nt = run_experiment_wound(params=params_no_tune, t_wound=t_wound, t_final=t_final)
    t_st, V_st, Y_st = run_experiment_wound(params=params, t_wound=t_wound, t_final=t_final)

    M_nt = extract_site_trajectory(Y_nt, site_idx=3)
    M_st = extract_site_trajectory(Y_st, site_idx=3)

    lambdas_nt = None
    lambdas_st = None
    if lyap_probe:
        control_zero = lambda t: np.zeros((N_SITES, N_TILES))  # noqa: E731
        x_nt, v_nt, r_nt, h1_nt, h4_nt, g1_nt, eta_nt = unpack_state(Y_nt[:, -1])
        x_st, v_st, r_st, h1_st, h4_st, g1_st, eta_st = unpack_state(Y_st[:, -1])
        lambdas_nt = estimate_lyapunov_per_tile(
            x_nt,
            v_nt,
            r_nt,
            h1_nt,
            h4_nt,
            g1_nt,
            eta_nt,
            params_no_tune,
            control_zero,
            T_total=200.0,
            T_chunk=2.0,
        )
        lambdas_st = estimate_lyapunov_per_tile(
            x_st,
            v_st,
            r_st,
            h1_st,
            h4_st,
            g1_st,
            eta_st,
            params,
            control_zero,
            T_total=200.0,
            T_chunk=2.0,
        )

    return {
        "no_tune": {"t": t_nt, "V": V_nt, "M": M_nt, "lambdas": lambdas_nt, "params": params_no_tune},
        "self_tune": {"t": t_st, "V": V_st, "M": M_st, "lambdas": lambdas_st, "params": params},
        "pattern": {"head_scale": head_scale, "tail_scale": tail_scale},
    }
