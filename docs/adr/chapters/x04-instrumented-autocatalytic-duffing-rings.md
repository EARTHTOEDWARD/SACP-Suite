# Chapter X04 — Instrumented ADRs and Cognitive Knees (Executable Appendix)

This draft extends the ADR line with an executable scaffold and three concrete experiments (network cognitive knee, autocatalytic gain, autopoietic recovery). It mirrors the “Self-Tuning ADRs and Cognitive Knees” chapter but is packaged as a ready-to-run appendix.

## 0. What this chapter is about

- Turn the ADR equations into a **programmable ring tile** (Duffing + resource loop + per-site plasticity).
- Define light **CLC-style metrics** for each site and the ring.
- Provide **three experiment helpers**: cognitive knee scan, autocatalytic vs. isolated, and lesion/autopoiesis.
- Keep it minimal (Euler integrator, no JIT) and easy to swap in RK4 later.

## 1. Appendix: Instrumented ADR Python scaffold

Save the code below as `ZZ_AUTOCATALYTIC DUFFING RINGS/experiments/instrumented_adr.py` (or similar). It is self-contained aside from `numpy` and `matplotlib`.

```python
"""
Instrumented Autocatalytic Duffing Rings (iADR)
================================================

Minimal simulation scaffold for the Instrumented ADR (iADR).
Features:
- Duffing-like sites x_i with velocities v_i
- Resource variables r_i with autocatalytic drive around the ring
- Per-site coupling strengths k_i(t) with simple plasticity
- Local/global coherence metrics
- CLC-style proxy metrics
- Three experiment helpers:
    * network cognitive knee
    * autocatalytic vs isolated
    * lesion / autopoietic recovery

Dependencies: numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# 1. Core iADR ring: dynamics, plasticity, logging
# --------------------------------------------------------------------

class InstrumentedADR:
    """
    Instrumented Autocatalytic Duffing Ring (iADR).

    State per site i:
        x[i] : position
        v[i] : velocity = dx/dt
        r[i] : resource / catalyst
        k[i] : diffusive coupling strength (plastic)

    Equations (Euler-integrated):

        ddot x_i = -gamma_i * v_i
                   + a_i * x_i - x_i^3
                   + k_i * Laplacian(x)_i
                   + eta * r_{i-1}
                   + F * cos(omega * t)

        dot r_i = -mu * r_i + sigma * x_i^2

    Plasticity (every `plasticity_every` steps, if enabled):

        k_i <- k_i + eta_plast * (R_target - R_local[i])
        clamp to [k_min, k_max]

    R_local[i] is a local phase-coherence metric.
    """

    def __init__(
        self,
        N=16,
        dt=0.01,
        a=1.0,
        gamma=0.2,
        mu=0.1,
        sigma=0.5,
        eta=0.5,
        F=0.0,
        omega=1.0,
        k_init=0.5,
        k_min=0.0,
        k_max=1.5,
        plasticity_enabled=True,
        eta_plast=1e-3,
        R_target=0.6,
        plasticity_every=10,
        neighborhood_radius=2,
        seed=None,
    ):
        self.N = N
        self.dt = float(dt)
        rng = np.random.default_rng(seed)

        self.a = np.full(N, a, float) if np.isscalar(a) else np.asarray(a, float)
        self.gamma = (
            np.full(N, gamma, float) if np.isscalar(gamma) else np.asarray(gamma, float)
        )

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.F = float(F)
        self.omega = float(omega)

        self.x = rng.uniform(-1.0, 1.0, size=N)
        self.v = rng.uniform(-0.1, 0.1, size=N)
        self.r = np.zeros(N, dtype=float)

        self.k = np.full(N, k_init, float)
        self.k_min = float(k_min)
        self.k_max = float(k_max)

        self.plasticity_enabled = bool(plasticity_enabled)
        self.eta_plast = float(eta_plast)
        self.R_target = float(R_target)
        self.plasticity_every = int(plasticity_every)
        self.neighborhood_radius = int(max(1, neighborhood_radius))

        self.t = 0.0
        self._step_counter = 0

    def _laplacian(self, x):
        return np.roll(x, -1) + np.roll(x, +1) - 2.0 * x

    def _update_state(self):
        dt = self.dt
        x, v, r = self.x, self.v, self.r
        lap = self._laplacian(x)
        r_prev = np.roll(r, +1)

        accel = (
            -self.gamma * v
            + self.a * x
            - x**3
            + self.k * lap
            + self.eta * r_prev
            + self.F * np.cos(self.omega * self.t)
        )

        v_new = v + dt * accel
        x_new = x + dt * v_new
        r_new = r + dt * (-self.mu * r + self.sigma * x**2)

        self.v = v_new
        self.x = x_new
        self.r = r_new

        self.t += dt
        self._step_counter += 1

    def _phases(self):
        eps = 1e-9
        return np.arctan2(self.v, self.x + eps)

    def coherence(self):
        theta = self._phases()
        z = np.exp(1j * theta)
        R_global = np.abs(np.mean(z))

        R_local = np.zeros(self.N, dtype=float)
        R = self.neighborhood_radius
        for i in range(self.N):
            idx = [(i + offset) % self.N for offset in range(-R, R + 1)]
            R_local[i] = np.abs(np.mean(z[idx]))
        return float(R_global), R_local

    def plasticity_update(self, R_local):
        if not self.plasticity_enabled:
            return
        self.k += self.eta_plast * (self.R_target - R_local)
        self.k = np.clip(self.k, self.k_min, self.k_max)

    def step(self):
        self._update_state()
        R_global, R_local = self.coherence()
        if (
            self.plasticity_enabled
            and (self._step_counter % max(1, self.plasticity_every) == 0)
        ):
            self.plasticity_update(R_local)
        return R_global, R_local

    def run(self, n_steps=5000, log_every=10):
        log_every = max(1, int(log_every))
        times, xs, vs, rs, ks = [], [], [], [], []
        Rg_list, Rl_list = [], []

        for step in range(n_steps):
            Rg, Rl = self.step()
            if step % log_every == 0:
                times.append(self.t)
                xs.append(self.x.copy())
                vs.append(self.v.copy())
                rs.append(self.r.copy())
                ks.append(self.k.copy())
                Rg_list.append(Rg)
                Rl_list.append(Rl.copy())

        return {
            "time": np.array(times),
            "x": np.vstack(xs),
            "v": np.vstack(vs),
            "r": np.vstack(rs),
            "k": np.vstack(ks),
            "R_global": np.array(Rg_list),
            "R_local": np.vstack(Rl_list),
        }


# --------------------------------------------------------------------
# 2. CLC-style metrics on logs
# --------------------------------------------------------------------

def compute_clc_proxy(
    log,
    dt,
    ac_max_lag=200,
    ac_threshold=1.0 / np.e,
    R_thresh=0.6,
    corr_thresh=0.3,
):
    """
    Cheap CLC-like proxy per site and for the ring.
    """
    X = np.asarray(log["x"])  # (T, N)
    R_local_hist = np.asarray(log["R_local"])  # (T, N)
    T, N = X.shape
    ac_max_lag = min(ac_max_lag, T - 1)

    # Identity lifetime
    C_inf = (R_local_hist > R_thresh).sum(axis=0) * dt

    # Autocorrelation horizon
    tau_past = np.zeros(N, dtype=float)
    for i in range(N):
        x_i = X[:, i] - X[:, i].mean()
        if np.allclose(x_i, 0.0):
            continue
        acf_full = np.correlate(x_i, x_i, mode="full")
        acf = acf_full[acf_full.size // 2:]
        if acf[0] == 0.0:
            continue
        acf /= acf[0]
        max_lag = min(ac_max_lag, len(acf) - 1)
        below = np.where(acf[1:max_lag + 1] < ac_threshold)[0]
        lag_idx = below[0] + 1 if below.size > 0 else max_lag
        tau_past[i] = lag_idx * dt
    tau_future = tau_past.copy()

    # Spatial radius from correlation matrix
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.corrcoef(Xc, rowvar=False) if T > 1 else np.eye(N)
    R_spatial = np.zeros(N, dtype=float)
    for i in range(N):
        corr_i = C[i]
        strong = np.where(np.abs(corr_i) > corr_thresh)[0]
        strong = strong[strong != i]
        if strong.size == 0:
            continue
        dists = np.abs(strong - i)
        dists = np.minimum(dists, N - dists)
        R_spatial[i] = dists.mean()

    eps = 1e-8
    S_node = C_inf * np.sqrt(tau_past * tau_future) * (R_spatial + eps)
    S_ring = float(np.mean(S_node))

    diag = {
        "C_inf": C_inf,
        "tau_past": tau_past,
        "tau_future": tau_future,
        "R_spatial": R_spatial,
    }
    return S_node, S_ring, diag


# --------------------------------------------------------------------
# 3. Experiment helpers
# --------------------------------------------------------------------

def experiment_network_knee(
    F_values,
    N=16,
    dt=0.01,
    n_steps=8000,
    log_every=10,
    seed=1,
    **adr_kwargs,
):
    """
    Sweep driving amplitude F and compute network CLC proxy.
    """
    S_ring_values = []
    for F in F_values:
        adr = InstrumentedADR(
            N=N,
            dt=dt,
            F=F,
            seed=seed,
            plasticity_enabled=False,
            **adr_kwargs,
        )
        log = adr.run(n_steps=n_steps, log_every=log_every)
        _, S_ring, _ = compute_clc_proxy(log, dt=dt)
        S_ring_values.append(S_ring)
    return np.array(F_values), np.array(S_ring_values)


def experiment_autocatalytic_vs_isolated(
    N=16,
    dt=0.01,
    n_steps_iso=8000,
    n_steps_ring=20000,
    log_every=10,
    seed=2,
    **adr_kwargs,
):
    """
    Compare CLC proxy in isolation vs coupled/plastic ring.
    """
    adr_iso = InstrumentedADR(
        N=N,
        dt=dt,
        k_init=0.0,
        eta=0.0,
        plasticity_enabled=False,
        seed=seed,
        **adr_kwargs,
    )
    log_iso = adr_iso.run(n_steps=n_steps_iso, log_every=log_every)
    S_iso_node, S_iso, _ = compute_clc_proxy(log_iso, dt=dt)

    adr_ring = InstrumentedADR(
        N=N,
        dt=dt,
        plasticity_enabled=True,
        seed=seed,
        **adr_kwargs,
    )
    log_ring = adr_ring.run(n_steps=n_steps_ring, log_every=log_every)
    S_ring_node, S_ring, _ = compute_clc_proxy(log_ring, dt=dt)

    return S_iso_node, S_ring_node, S_iso, S_ring


def experiment_lesion_autopoiesis(
    lesion_sites,
    N=16,
    dt=0.01,
    n_steps_pre=15000,
    n_steps_post=15000,
    log_every=10,
    seed=3,
    **adr_kwargs,
):
    """
    Train with plasticity, lesion some sites (zero k_i), continue and see if CLC recovers.
    """
    lesion_sites = np.array(lesion_sites, dtype=int)

    adr = InstrumentedADR(
        N=N,
        dt=dt,
        plasticity_enabled=True,
        seed=seed,
        **adr_kwargs,
    )
    log_pre = adr.run(n_steps=n_steps_pre, log_every=log_every)
    _, S_pre, _ = compute_clc_proxy(log_pre, dt=dt)

    adr.k[lesion_sites] = 0.0

    log_post = adr.run(n_steps=n_steps_post, log_every=log_every)
    _, S_post, _ = compute_clc_proxy(log_post, dt=dt)

    return log_pre, log_post, S_pre, S_post


# --------------------------------------------------------------------
# 4. Small demo when run as a script
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: network knee vs F
    F_vals = np.linspace(0.0, 0.8, 10)
    F_vals, S_vals = experiment_network_knee(
        F_vals,
        N=16,
        dt=0.01,
        n_steps=6000,
        log_every=10,
        a=1.0,
        gamma=0.2,
        mu=0.1,
        sigma=0.5,
        eta=0.5,
        k_init=0.4,
    )
    plt.figure()
    plt.plot(F_vals, S_vals, marker="o")
    plt.xlabel("Drive amplitude F")
    plt.ylabel("Network CLC proxy")
    plt.title("iADR network cognitive knee (proxy)")
    plt.tight_layout()

    # Example 2: autocatalytic vs isolated
    S_iso_node, S_ring_node, S_iso, S_ring = experiment_autocatalytic_vs_isolated(
        N=16,
        dt=0.01,
        n_steps_iso=6000,
        n_steps_ring=15000,
        log_every=10,
        a=1.0,
        gamma=0.2,
        mu=0.1,
        sigma=0.5,
        eta=0.5,
        k_init=0.4,
    )
    print("Network CLC in isolation :", S_iso)
    print("Network CLC in ring      :", S_ring)
    print("Fraction of sites improved:",
          np.mean(S_ring_node > S_iso_node))

    # Example 3: lesion & autopoiesis
    lesion_sites = list(range(0, 4))
    log_pre, log_post, S_pre, S_post = experiment_lesion_autopoiesis(
        lesion_sites,
        N=16,
        dt=0.01,
        n_steps_pre=15000,
        n_steps_post=15000,
        log_every=10,
        a=1.0,
        gamma=0.2,
        mu=0.1,
        sigma=0.5,
        eta=0.5,
        k_init=0.4,
    )
    print("CLC pre-lesion :", S_pre)
    print("CLC post-lesion:", S_post)

    plt.show()
```
