"""Cognitive metrics: memory profiles and discriminability."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sacp_suite.modules.sacp_x.metrics import rosenstein_lle


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x


def _discretize(values: np.ndarray, n_bins: int = 16) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if np.allclose(v.min(), v.max()):
        return np.zeros_like(v, dtype=int)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(v, qs)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.zeros_like(v, dtype=int)

    codes = np.digitize(v, edges[1:-1], right=True)
    return codes.astype(int)


def _mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    x_vals, x_codes = np.unique(x, return_inverse=True)
    y_vals, y_codes = np.unique(y, return_inverse=True)

    n_x = x_vals.size
    n_y = y_vals.size

    counts = np.zeros((n_x, n_y), dtype=float)
    for xi, yi in zip(x_codes, y_codes):
        counts[xi, yi] += 1.0

    total = counts.sum()
    if total == 0:
        return 0.0

    p_xy = counts / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    mask = p_xy > 0
    denom = (p_x @ p_y)
    ratio = np.zeros_like(p_xy)
    ratio[mask] = p_xy[mask] / denom[mask]

    mi = (p_xy[mask] * np.log2(ratio[mask])).sum()
    return float(max(mi, 0.0))


def compute_memory_profile(
    inputs: np.ndarray,
    outputs: np.ndarray,
    max_lag: int,
    n_output_bins: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    U = _ensure_2d(inputs)
    Y = np.asarray(outputs)
    if Y.ndim == 1:
        Y = Y[None, :, None]
    elif Y.ndim == 2:
        Y = Y[:, :, None]

    n_trials, T = U.shape
    _, Ty, d = Y.shape
    if Ty != T:
        raise ValueError("inputs and outputs must have same time length")
    if d < 1:
        raise ValueError("outputs must have at least one dimension")

    y_scalar = Y[:, :, 0]
    ks = np.arange(1, max_lag + 1)
    M_vals = np.zeros_like(ks, dtype=float)

    for idx, k in enumerate(ks):
        if k >= T:
            break

        u_lags = []
        y_now = []
        for tr in range(n_trials):
            u = U[tr]
            y = y_scalar[tr]
            u_lags.append(u[:-k])
            y_now.append(y[k:])

        u_lags_arr = np.concatenate(u_lags, axis=0)
        y_now_arr = np.concatenate(y_now, axis=0)

        if np.issubdtype(U.dtype, np.integer):
            u_codes = u_lags_arr.astype(int)
        else:
            u_codes = _discretize(u_lags_arr, n_bins=16)

        y_codes = _discretize(y_now_arr, n_bins=n_output_bins)
        M_vals[idx] = _mutual_information_discrete(u_codes, y_codes)

    return ks, M_vals


def _mean_autocorr(x: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return lags and average absolute autocorrelation across dimensions."""

    if x.ndim == 1:
        x = x[:, None]
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0) + 1e-9
    lags = np.arange(1, max_lag + 1)
    ac = np.zeros_like(lags, dtype=float)
    for idx, k in enumerate(lags):
        if k >= x.shape[0]:
            break
        prod = x[:-k] * x[k:]
        ac[idx] = float(np.mean(np.abs(np.sum(prod, axis=1) / np.sum(var))))
    return lags, ac


def estimate_clc_metrics(
    states: np.ndarray,
    dt: float = 1.0,
    capture_sigma: float = 1.0,
    autocorr_thresh: float = 0.1,
    max_lag: int | None = None,
) -> Dict[str, float]:
    """Estimate a lightweight Cognitive Light Cone for a trajectory.

    Parameters
    ----------
    states: np.ndarray
        Array of shape (T, d) holding the trajectory (post burn-in).
    dt: float
        Timestep between samples.
    capture_sigma: float
        Width multiplier for the capture band around the mean state norm.
    autocorr_thresh: float
        Threshold where autocorrelation is considered to have "forgotten" the past.
    max_lag: int | None
        Maximum lag to evaluate; defaults to min(T//2, 400).
    """

    X = np.asarray(states, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T = X.shape[0]
    if T < 10:
        return {"tau_past": 0.0, "tau_future": 0.0, "radius": 0.0, "capture": 0.0, "clc": 0.0}

    max_lag = max_lag or min(T // 2, 400)
    lags, ac = _mean_autocorr(X, max_lag=max_lag)
    below = np.where(ac < autocorr_thresh)[0]
    tau_past_steps = lags[below[0]] if below.size else lags[-1]
    tau_past = float(tau_past_steps * dt)

    # Largest Lyapunov exponent as a rough future horizon proxy.
    lam = rosenstein_lle(X.reshape(-1), m=6, tau=4)
    lam = float(lam / max(dt, 1e-9))
    tau_future = float(1.0 / max(lam, 1e-6)) if lam > 0 else float(tau_past)

    norms = np.linalg.norm(X - np.mean(X, axis=0, keepdims=True), axis=1)
    radius = float(np.sqrt(np.mean(norms**2)))
    band = np.mean(norms) + capture_sigma * np.std(norms)
    capture = float(np.mean(norms <= band)) if band > 0 else 0.0

    clc_score = float(capture * np.sqrt(max(tau_past, 0.0) * max(tau_future, 0.0) * max(radius, 0.0)))

    return {
        "tau_past": tau_past,
        "tau_future": tau_future,
        "radius": radius,
        "capture": capture,
        "clc": clc_score,
    }


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def estimate_discriminability(
    labels: np.ndarray,
    trajectories: np.ndarray,
    test_size: float = 0.3,
    random_state: int | None = 0,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    X = np.asarray(trajectories)
    if X.ndim == 2:
        X = X[:, :, None]

    n_trials, T, d = X.shape
    X_flat = X.reshape(n_trials, T * d)

    if labels.shape[0] != n_trials:
        raise ValueError("labels and trajectories must have same n_trials")

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    Pe = 1.0 - acc

    unique_labels = np.unique(labels)
    K = unique_labels.size
    if K < 2:
        return {"K": float(K), "T": float(T), "accuracy": float(acc), "Pe": float(Pe), "I_lower": 0.0, "D": 0.0}

    H_C = np.log2(K)
    I_lower = H_C - _binary_entropy(Pe) - Pe * np.log2(K - 1)
    I_lower = max(I_lower, 0.0)
    D_val = I_lower / H_C if H_C > 0 else 0.0
    D_val = float(np.clip(D_val, 0.0, 1.0))

    return {
        "K": float(K),
        "T": float(T),
        "accuracy": float(acc),
        "Pe": float(Pe),
        "I_lower": float(I_lower),
        "D": D_val,
    }
