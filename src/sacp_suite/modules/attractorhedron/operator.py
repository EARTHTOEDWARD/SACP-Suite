"""Attractorhedron operator utilities."""

from __future__ import annotations

import numpy as np
from numpy.linalg import eig


def build_ulam(points: np.ndarray, nx: int = 25, ny: int = 25, bounds: tuple[float, float, float, float] | None = None):
    """Build a row-stochastic Ulam operator from 2D samples (x, z)."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be (N,2) array")
    x, z = pts[:, 0], pts[:, 1]

    if bounds is None:
        qmin, qmax = 0.01, 0.99
        xmin, xmax = np.quantile(x, qmin), np.quantile(x, qmax)
        zmin, zmax = np.quantile(z, qmin), np.quantile(z, qmax)
    else:
        xmin, xmax, zmin, zmax = bounds

    xi = np.clip(((x - xmin) / (xmax - xmin) * nx).astype(int), 0, nx - 1)
    zi = np.clip(((z - zmin) / (zmax - zmin) * ny).astype(int), 0, ny - 1)

    i_idx = zi[:-1] * nx + xi[:-1]
    j_idx = zi[1:] * nx + xi[1:]
    N = nx * ny
    counts = np.zeros((N, N), dtype=float)
    for a, b in zip(i_idx, j_idx, strict=False):
        counts[a, b] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    # rows with no observed transitions should stay identically zero after normalization
    P = np.divide(
        counts,
        np.maximum(row_sums, 1e-12),
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )
    bounds_out = (xmin, xmax, zmin, zmax)
    return P, bounds_out


def analyze_operator(P: np.ndarray, mean_return: float = 1.0, nx: int = 25, ny: int = 25) -> dict:
    """Return spectral stats for a row-stochastic matrix."""

    mat = np.asarray(P, dtype=float)
    # Guard against degenerate/zero rows: eigenvalues of a non-negative
    # row-stochastic matrix should live in the unit disk; numeric blow-ups
    # can appear when rows are all zero. Clamp to keep metrics bounded.
    w, V = eig(mat.T)
    order = np.argsort(-np.abs(w))
    w, V = w[order], V[:, order]
    lam2 = w[1] if len(w) > 1 else 0.0
    if not np.isfinite(lam2):
        lam2 = 0.0
    abs_l2 = float(np.clip(np.abs(lam2), 0.0, 1.0))
    gamma = float(-np.log(max(abs_l2, 1e-12)) / max(mean_return, 1e-12))
    v2 = np.real(V[:, 1]).reshape(ny, nx).tolist() if mat.size else []
    return {
        "lambda2": [float(np.real(lam2)), float(np.imag(lam2))],
        "abs_lambda2": abs_l2,
        "gamma": gamma,
        "v2_field": v2,
    }


def left_right_gate(P: np.ndarray, nx: int, ny: int, alpha: float = 1.0) -> np.ndarray:
    """Attenuate cross-half transitions by factor alpha."""

    mat = np.asarray(P, dtype=float)
    N = nx * ny
    if mat.shape != (N, N):
        raise ValueError("Matrix shape mismatch for supplied nx, ny")

    x_centers = (np.arange(nx) + 0.5)
    left_mask = x_centers < (nx / 2.0)
    left_indices = np.where(np.repeat(left_mask, ny))[0]
    right_indices = np.setdiff1d(np.arange(N), left_indices)

    gated = mat.copy()
    for i in left_indices:
        gated[i, right_indices] *= alpha
    for i in right_indices:
        gated[i, left_indices] *= alpha

    row_sums = gated.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        gated = np.divide(gated, row_sums, where=row_sums > 0)
    gated[row_sums.squeeze() == 0] = 0.0
    return gated
