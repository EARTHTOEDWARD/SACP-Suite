"""Canonical 4-site ADR–Krebs tile built on InstrumentedADR."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Optional, Sequence, Tuple, Dict

import numpy as np

from .instrumented_adr import InstrumentedADR, compute_clc_proxy

# ---------------------------------------------------------------------
# 1. Config for ADR–Krebs
# ---------------------------------------------------------------------


@dataclass
class KrebsADRConfig:
    # integration
    dt: float = 0.01

    # Duffing stiffness parameters per site (V_i(x) = x^4/4 - a_i x^2 / 2)
    a_TCA: float = 1.10
    a_NADH: float = 1.00
    a_dPsi: float = 1.20
    a_ATP: float = 0.90

    # Damping per site
    gamma_TCA: float = 0.22
    gamma_NADH: float = 0.20
    gamma_dPsi: float = 0.18
    gamma_ATP: float = 0.16

    # Resource dynamics shared across sites
    mu: float = 0.15  # resource decay ("leak")
    sigma: float = 0.60  # resource production from x_i^2

    # Base catalytic drive around the ring
    eta_base: float = 0.50

    # External periodic drive (usually off)
    F: float = 0.0
    omega: float = 1.0

    # Diffusive coupling initial value k_i(0)
    k_init: float = 0.40
    k_min: float = 0.0
    k_max: float = 1.5

    # Plasticity options (off by default)
    plasticity_enabled: bool = False
    eta_plast: float = 1e-3
    R_target: float = 0.6
    plasticity_every: int = 10
    neighborhood_radius: int = 1

    seed: Optional[int] = None


# ---------------------------------------------------------------------
# 2. Factory: build a 4-site ADR ring specialised for Krebs
# ---------------------------------------------------------------------

SITE_NAMES = ["TCA_flux", "NADH_ratio", "DeltaPsi", "ATP_ratio"]


def make_krebs_adr(alpha: float = 1.0, cfg: Optional[KrebsADRConfig] = None, **overrides) -> InstrumentedADR:
    """
    Build an InstrumentedADR configured as an ADR–Krebs surrogate.

    Parameters
    ----------
    alpha : float
        Dimensionless control parameter ~ energy demand / ATP dissipation.
        alpha = 1.0 is a reference regime.
    cfg : KrebsADRConfig or None
        Optional config; if None, uses defaults.
    overrides : dict
         Optional overrides for KrebsADRConfig fields (e.g., dt, seed).
    """
    if cfg is None:
        cfg = KrebsADRConfig()
    if overrides:
        cfg = dataclass_replace(cfg, **overrides)

    a = np.array([cfg.a_TCA, cfg.a_NADH, cfg.a_dPsi, cfg.a_ATP], dtype=float)
    gamma = np.array([cfg.gamma_TCA, cfg.gamma_NADH, cfg.gamma_dPsi, cfg.gamma_ATP], dtype=float)

    adr = InstrumentedADR(
        N=4,
        dt=cfg.dt,
        a=a,
        gamma=gamma,
        mu=cfg.mu,
        sigma=cfg.sigma,
        eta=cfg.eta_base * alpha,  # alpha is our "energy demand" knob
        F=cfg.F,
        omega=cfg.omega,
        k_init=cfg.k_init,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        plasticity_enabled=cfg.plasticity_enabled,
        eta_plast=cfg.eta_plast,
        R_target=cfg.R_target,
        plasticity_every=cfg.plasticity_every,
        neighborhood_radius=cfg.neighborhood_radius,
        seed=cfg.seed,
    )

    # Metadata for dashboards
    adr.site_names = SITE_NAMES
    adr.meta = {
        "model": "ADR-Krebs-v1",
        "description": "4-site Autocatalytic Duffing Ring surrogate for Krebs",
        "mapping": {
            "x[0]": "TCA throughput",
            "x[1]": "NADH / NAD+ ratio",
            "x[2]": "DeltaPsi (membrane potential)",
            "x[3]": "ATP / ADP ratio",
        },
        "control_parameter": "alpha (eta gain ~ energy demand / ATP dissipation)",
    }
    return adr


# ---------------------------------------------------------------------
# 3. Helpers: run trajectories & compute diagnostics
# ---------------------------------------------------------------------


def run_krebs_trajectory(
    alpha: float,
    n_steps: int = 20000,
    log_every: int = 10,
    cfg: Optional[KrebsADRConfig] = None,
) -> tuple[InstrumentedADR, Dict[str, np.ndarray], float, Dict[str, float]]:
    """
    Convenience wrapper:
      - build ADR–Krebs tile for given alpha
      - integrate for n_steps
      - return logs + ring CLC proxy
    """
    if cfg is None:
        cfg = KrebsADRConfig()
    adr = make_krebs_adr(alpha=alpha, cfg=cfg)
    log = adr.run(n_steps=n_steps, log_every=log_every)
    S_node, S_ring, diag = compute_clc_proxy(log, dt=cfg.dt)
    diag = {**diag, "S_node_mean": float(np.mean(S_node)), "S_node": S_node.tolist()}
    return adr, log, S_ring, diag


def krebs_bifurcation_scan(
    alpha_values: Sequence[float],
    n_steps_transient: int = 20000,
    n_steps_sample: int = 20000,
    sample_every: int = 10,
    cfg: Optional[KrebsADRConfig] = None,
    sample_site: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a Grytsay-style bifurcation diagram for ADR–Krebs by sweeping alpha.

    Protocol (per alpha):
      - create ADR–Krebs ring
      - run for n_steps_transient (discard)
      - run for n_steps_sample, logging every sample_every
      - collect samples of x[:, sample_site] (default TCA throughput)
    """
    if cfg is None:
        cfg = KrebsADRConfig()

    all_alphas = []
    x_samples = []

    for alpha in alpha_values:
        adr = make_krebs_adr(alpha=alpha, cfg=cfg)

        # burn-in / transient
        _ = adr.run(n_steps=n_steps_transient, log_every=max(1, n_steps_transient))

        # sample on attractor
        log = adr.run(n_steps=n_steps_sample, log_every=sample_every)
        X = np.asarray(log["x"])  # shape (T, 4)
        x_site = X[:, sample_site]

        all_alphas.extend([alpha] * len(x_site))
        x_samples.extend(x_site.tolist())

    return np.asarray(all_alphas), np.asarray(x_samples)
