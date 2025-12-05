# Chapter 7 — ADR–Krebs: A Canonical Autocatalytic Tile for Metabolism

## 7.0 Why this chapter exists

Up to now we’ve built three strands:

* **Grytsay-style metabolic chaos** — detailed ODE models of oxidative metabolism and the Krebs cycle that show Feigenbaum cascades and strange attractors as ATP dissipation is varied.
* **MAES / MAES-ST** — a categorical, topos-theoretic view of “agent–environment” systems with horizontal/vertical/strange-attractor morphisms and context sheaves over parameters like \(\alpha_4\) (ATP dissipation).
* **Autocatalytic Duffing Rings (ADRs)** — cyclic networks of Duffing-like oscillators with resource loops that self-organise on high-dimensional strange attractors and can be instrumented, self-tuned, and reused as “tiles” in larger architectures.

This chapter glues those strands together into something very practical:

> A canonical ADR–Krebs tile: a 4-site Autocatalytic Duffing Ring that mimics the attractor structure of the Krebs cycle (and similar autocatalytic chemistries) and can be dropped into the SACP suite as a standard module.

We’ll:

* Recap the Krebs-chaos story in Grytsay’s models.
* Give the MAES-ST reading: \(\alpha_4\) as a context coordinate and chaotic regimes as strange-attractor morphisms.
* Build a 4-site ADR ring whose variables map to core metabolic coordinates:
  * \(x_1\): TCA throughput,
  * \(x_2\): NADH/NAD\(^+\),
  * \(x_3\): mitochondrial potential \(\Delta\Psi\),
  * \(x_4\): ATP/ADP.
* Define the ADR–Krebs equations and discuss how to tune a drive/leak parameter so its bifurcation diagram qualitatively matches Grytsay’s.
* Show how this becomes a drop-in SACP model and a template for other chemistries (hemostasis, glycolysis, GRNs).

This is not meant as a perfect biochemical fit; it’s a canonical toy metabolon: compact, chaotic, reusable.

## 7.1 Grytsay’s Krebs cycle as a strange attractor

Grytsay and collaborators model the Krebs cycle plus oxidative phosphorylation with a high-dimensional ODE system (10–20 variables, depending on version) that explicitly tracks substrates, NADH, membrane potential, ATP, and regulatory feedbacks.

Key observations:

* There is a control parameter \(\alpha_4\), representing ATP dissipation / energy turnover.
* As \(\alpha_4\) is varied, the system undergoes a period-doubling cascade: steady state → limit cycle → 2-cycle → 4-cycle → … → chaos.
* In the chaotic regime:
  * trajectories stay bounded (homeodynamic rather than exploding),
  * Lyapunov exponents are positive,
  * projections onto metabolic planes like (ATP, NADH) and (\(\Delta\Psi\), NADH) show the familiar fractal strange-attractor structure.

So the Krebs cycle is not just a “loop of reactions”; it’s a context-dependent attractor bundle:

* For some \(\alpha_4\): a quiet fixed point (resting metabolism).
* For others: oscillatory behaviour (metabolic rhythms).
* For yet others: chaotic exploration (stress, adaptation, brink of failure).

The biology doesn’t particularly care which ODE terms produced this, only that the geometry of the attractor and its dependence on \(\alpha_4\) are real and reusable.

## 7.2 The MAES-ST reading of metabolic chaos

In MAES-ST we describe a system as a sheaf of MAES configurations over a base space of contexts \(B\).

For metabolism:

* **Base space \(B\)**: contexts like (oxygen, substrate, ATP demand). \(\alpha_4\) lives here.
* **\(M(U)\) for a context \(U \subset B\)**:
  * Agent \(A(U)\): internal metabolic variables (intermediates, NADH, \(\Delta\Psi\), ATP).
  * Environment \(E(U)\): rest of the cell and tissues (nutrient supply, workload).
  * Blanket \(B(U)\): interfaces (membrane transporters, ETC, ATPases).
  * Interaction dynamics \(I(U)\): local ODEs/SDEs derived from biochemistry.
* **Horizontal morphisms \(h\)**: time evolution within a fixed \(\alpha_4\) (normal metabolic dynamics).
* **Vertical morphisms \(v\)**: shifts in \(\alpha_4\) or other parameters (changing workload, hypoxia).
* **Strange-attractor morphisms \(\eta_{\mathrm{SA}}\)**: the chaotic regimes where the system explores a fractal attractor at fixed \(\alpha_4\).

In this view, Grytsay’s period-doubling cascade is a family of vertical morphisms in the context topos, and the chaotic Krebs cycle is a canonical example of an \(\eta_{\mathrm{SA}}\) morphism in a biochemical MAES tile.

The question this chapter answers is:

> Can we compress that complex MAES–Krebs object into a small ADR tile that lives in the same morphism class (same qualitative attractor anatomy vs a drive parameter)?

That’s what ADR–Krebs is for.

## 7.3 Recap: Autocatalytic Duffing Rings as canonical tiles

Earlier we defined an ADR as a ring of nonlinear oscillators \(x_i\) with resource loops \(r_i\):

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i(x_i) = \kappa (x_{i+1} + x_{i-1} - 2 x_i) + \eta r_{i-1} + \xi_i(t), \tag{7.1}
\]
\[
\dot r_i = -\mu r_i + \sigma x_i^2. \tag{7.2}
\]

Each site has a Duffing-like potential \(V_i\) (often double-well).

* \(\kappa\): diffusive coupling along the ring.
* \(\eta\): autocatalytic drive from the previous site’s resource.
* \(r_i\): slow “resource” that accumulates activity and decays.
* \(\mu\): resource leak / dissipation; \(\sigma\): production.

Even for modest \(N\), these rings:

* exhibit limit cycles, travelling waves, and high-dimensional strange attractors;
* can be instrumented with Lyapunov spectra and CLC-style metrics;
* can be wrapped in self-tuning gates that keep them near the edge of chaos.

In other words, an ADR is already a minimal autocatalytic strange-attractor machine. To turn it into ADR–Krebs, we just need to:

* pick four meaningful sites;
* shape their potentials and couplings to match metabolic roles;
* pick a parameter (like \(\mu\)) to play the role of effective \(\alpha_4\).

## 7.4 The 4-site ADR–Krebs ring

We now define a 4-site ADR ring, with states \(x_i, v_i = \dot x_i, r_i\), indexed modulo 4:

* \(x_1\): TCA throughput (coarse flux around the Krebs loop).
* \(x_2\): NADH/NAD\(^+\) ratio.
* \(x_3\): membrane potential \(\Delta\Psi\).
* \(x_4\): ATP/ADP ratio.

Think of each \(x_i\) as a coarse “mode” (normalised, dimensionless), not a literal concentration.

### 7.4.1 Site potentials: metabolic “modes”

Give each site a double-well potential:

\[
V_i(x_i) = \tfrac14 x_i^4 - \tfrac{a_i}{2} x_i^2. \tag{7.3}
\]

Low-activity and high-activity wells correspond to different metabolic modes:

* For \(x_1\): low vs high TCA flux.
* For \(x_2\): reduced vs oxidised NAD pool.
* For \(x_3\): low vs high \(\Delta\Psi\).
* For \(x_4\): low vs high ATP/ADP.

As in standard Duffing, \(a_i > 0\) gives a proper double-well; heterogeneity in \(a_i\) can encode tissue- or cell-type differences.

The second-order dynamics per site:

\[
\dot x_i = v_i, \qquad
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \text{coupling} + \text{drive}. \tag{7.4}
\]

### 7.4.2 Resource loops as coarse biochemical fluxes

We keep the resource form from the generic ADR:

\[
\dot r_i = -\mu r_i + \sigma x_i^2. \tag{7.5}
\]

Interpretation (coarse):

* \(r_1\): cumulative flux of carbon / reducing equivalents generated by TCA.
* \(r_2\): cumulative reducing power feeding the ETC.
* \(r_3\): integrated proton-motive force / charge separation history.
* \(r_4\): accumulated ATP-equivalent “spend”.

We then choose a metabolic ordering for the autocatalytic drive:

\[
\dot v_1 \;\ni\; +\eta_{41} r_4 \quad \text{(ATP demand drives TCA)}, \\
\dot v_2 \;\ni\; +\eta_{12} r_1 \quad \text{(TCA throughput drives NADH)}, \\
\dot v_3 \;\ni\; +\eta_{23} r_2 \quad \text{(NADH drives \(\Delta\Psi\))}, \\
\dot v_4 \;\ni\; +\eta_{34} r_3 \quad \text{(\(\Delta\Psi\) drives ATP production)}. \tag{7.6}
\]

This gives an explicit four-step autocatalytic ring:

* \( \text{TCA} \to \text{NADH} \to \Delta\Psi \to \text{ATP} \to \text{back to TCA}. \)

We may also include weak diffusive coupling between neighbouring \(x_i\) to represent cross-talk:

\[
\kappa (x_{i+1} + x_{i-1} - 2 x_i). \tag{7.7}
\]

### 7.4.3 Full ADR–Krebs equations

Collecting everything:

\[
\dot x_i = v_i,
\]
\[
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \kappa (x_{i+1} + x_{i-1} - 2 x_i) + \eta_{i-1,i} r_{i-1} + F_i(t),
\]
\[
\dot r_i = -\mu r_i + \sigma x_i^2, \qquad i \in \{1,2,3,4\} \ (\text{mod } 4). \tag{7.8}
\]

where:

* \(F_i(t)\) are optional external drives (e.g., workload pulses).
* Parameters \(\gamma_i, a_i, \kappa, \eta_{i-1,i}, \mu, \sigma\) are chosen to match the desired attractor anatomy.

Canonical choices (v1 ADR–Krebs):

* Mild heterogeneity:
  * \(a_1 = a_3 = 1.0\) (more bistable), \(a_2 = a_4 = 0.8\).
  * \(\gamma_i \in [0.15, 0.25]\).
* Ring couplings:
  * \(\kappa \approx 0.05\) (enough to correlate sites, not enough to lock them).
  * \(\eta_{i-1,i} = \eta\) shared around the ring.
* Resource parameters:
  * \(\sigma \approx 0.5\).
  * \(\mu\) is our effective \(\alpha_4\) — higher \(\mu\) = faster dissipation / higher metabolic load.

We then scan \(\mu\) in a range (say \(\mu \in [0.01, 0.05]\)) and inspect:

* Poincaré slices of \((x_2, x_4)\) (NADH vs ATP),
* bifurcation diagrams of \(x_4\) vs \(\mu\),
* largest Lyapunov exponent vs \(\mu\).

Qualitatively, we aim for the same pattern seen in the detailed Krebs model:

* large \(\mu\): quiescent or simple limit cycles,
* intermediate \(\mu\): period-doubling of rhythmic ATP dynamics,
* lower \(\mu\): fully chaotic ATP/\(\Delta\Psi\)/NADH attractor, still bounded.

The exact \(\mu\) values are empirical; the target is the shape of the bifurcation diagram and the presence of a Feigenbaum-like cascade, not a literal parameter match.

## 7.5 MAES-ST classification of ADR–Krebs

Once ADR–Krebs is defined, it can be treated as a MAES–ST object just like the biochemical Krebs cycle:

* **Agent \(A\)**: the 4-site ADR ring state \(X = (x_i, v_i, r_i)\).
* **Environment \(E\)**: external workload and nutrient context (in SACP: the outer simulation).
* **Blanket \(B\)**: the interface variables we expose (e.g., \((x_1, x_4)\) as “input demand” and “output ATP”).
* **Context coordinate**: \(\mu\) (and possibly external drive statistics).
* **Horizontal morphisms** are just trajectories under (7.8) at fixed \(\mu\); **vertical morphisms** are \(\mu\)-shifts; **\(\eta_{\mathrm{SA}}\) morphisms** are the chaotic regimes.

The point of ADR–Krebs is that:

* It lives in the same morphism family as the real Krebs cycle in the MAES–ST sense.
* It is much smaller and computationally cheap, so we can embed dozens or hundreds in larger tissues, or use it in closed-loop control experiments.

This makes it a canonical metabolic MAES tile.

## 7.6 Wiring ADR–Krebs into the SACP suite

In the SACP / ABCP architecture, each model is a module with a Sense–Act–Couple–Perturb interface. The instrumented ADR scaffold from the executable appendix already implements this for general rings.

For ADR–Krebs, we specialise:

* **Sense**
  * Export time series of \(x_1, x_2, x_3, x_4\).
  * Provide ring-level diagnostics: Lyapunov proxies, CLC metrics, energy-like integrals.
  * Optionally expose “macro observables” like average ATP over windows, variability, etc.
* **Act**
  * Allow external modules to:
    * perturb \(F_1(t)\) (TCA input / substrate),
    * perturb \(F_4(t)\) or \(\mu\) (ATP demand shocks),
    * modulate \(\eta\) (coupling between NADH and \(\Delta\Psi\)).
* **Couple**
  * ADR–Krebs can be chained to other ADR tiles, Potts blankets, or GRN modules:
    * e.g., couple \(x_4\) to a gene-regulatory ADR tile controlling mitochondrial biogenesis, as in the GRN-as-MAES work.
* **Perturb**
  * Provide standard perturbation hooks for:
    * instantaneous kicks to \(x_i\),
    * parameter jumps in \(\mu\) or \(\gamma\) to mimic ischemia/reperfusion,
    * structural changes (turn off a link \(\eta_{23}\) to mimic ETC inhibition).

Because the ADR code is already implemented as a reusable Python class in the instrumented ADR module, ADR–Krebs is mainly a parameter preset plus a mapping table:

| ADR–Krebs site | Variable \(x_i\) | Biological interpretation          |
| --- | --- | --- |
| 1 | \(x_1\) | TCA flux / throughput |
| 2 | \(x_2\) | NADH/NAD\(^+\) redox state |
| 3 | \(x_3\) | membrane potential \(\Delta\Psi\) |
| 4 | \(x_4\) | ATP/ADP energy charge |

Given this, an LLM-based lab assistant or a human user can:

* Choose a \(\mu\)-range to sweep as a stand-in for \(\alpha_4\).
* Run the network-knee experiment tools (chapter 19/X04) directly on ADR–Krebs:
  * Observe a metabolic cognitive knee: drive too low → dead metabolism; drive just right → rich, structured fluctuations; drive too high → chaotic meltdown.
  * Compare the ADR–Krebs bifurcation diagrams to Grytsay’s ODEs for sanity.

## 7.7 From ADR–Krebs to a family of autocatalytic chemistry tiles

Once we have this one canonical tile, cloning the pattern is straightforward:

* **ADR–Hemostasis**: 4–6 sites representing thrombin burst, platelet activation, fibrin mesh, fibrinolysis, mapped onto an ADR ring with a drive/leak parameter playing the role of Grytsay’s \(\mu_0\) in the hemostasis model.
* **ADR–Glycolysis**: 3–4 sites representing phosphofructokinase activity, NADH and pyruvate, lactate / pH feedback.
* **ADR–GRN tiles**: using the GRN motifs (positive feedback, toggle switches, pluripotency core) as local “chemistry” and wrapping them with ADR-style resources, letting ATP and cellular state variables act as the drive.

Each tile then:

* lives in the same MAES–ST category (horizontal/vertical/\(\eta_{\mathrm{SA}}\) morphisms);
* can be instrumented with the same SACP dashboard (Lyapunov bands, CLCs, cluster oracles at the boundary);
* can be coupled into fractal blankets and small-world agent ecologies (chapters 20–23).

ADR–Krebs is simply the first named member of this family: the metabolic archetype that anchors the rest.

## 7.8 What “understanding this chapter” buys you

If you grok ADR–Krebs, you get three things:

1. **A mental compression**: “Metabolic chaos” is not a mysterious feat of a big biochemical ODE; it’s a particular shape of attractor that a 4-site autocatalytic Duffing ring can also realise.
2. **A practical module**: You can now drop ADR–Krebs into any SACP experiment as “a little metabolic brain” that:
   * responds nonlinearly to load,
   * shows bifurcations and chaos vs a drive parameter,
   * and can be kept near the edge of chaos by self-tuning gates.
3. **A template**: Any autocatalytic chemistry you care about — hemostasis, glycolysis, GRNs, signalling networks — can be similarly compressed into an ADR tile with:
   * 3–6 sites,
   * resource loops,
   * a chosen drive/leak parameter,
   * and a mapping table from \(x_i\) to chemical roles.

The rest of the book will lean heavily on that idea: “tile the universe with little ADR rings, then let them talk through self-tuning blankets.” ADR–Krebs is the first tile on the board.

---

## Appendix — Canonical ADR–Krebs tile (worked parameter set and mappings)

This appendix pins down a simulation-ready ADR–Krebs tile that mirrors Grytsay-style Krebs chaos with a single effective drive/leak parameter.

### A.1 Site and resource mapping (Krebs → ADR)

| ADR site | \(x_i\) meaning (dimensionless)          | Biological mapping (Grytsay vars)                       |
| --- | --- | --- |
| 1 | TCA throughput | \(S_3 + S_8\) (scaled) – overall Krebs flux |
| 2 | NADH/NAD\(^+\) redox | NADH fraction from (NADH + NAD\(^+\)=L\_2), using N |
| 3 | \(\Delta\Psi\) | membrane potential \(\psi\) (eq. 16) |
| 4 | ATP/ADP energy charge | ATP fraction from (ATP+ADP=L\_1), using T |

Suggested dimensionless definitions: subtract a reference steady state, e.g.
\(x_1 \approx \frac{S_3+S_8}{S_{\text{ref}}}-1\), \(x_2 \approx \frac{\text{NADH}}{L_2} - (\cdot)_{\text{ref}}\), \(x_3 \approx \frac{\psi}{\psi_{\text{ref}}}-1\), \(x_4 \approx \frac{\text{ATP}}{L_1} - (\cdot)_{\text{ref}}\).

Resource loop semantics:

| Resource | Meaning | Grytsay process |
| --- | --- | --- |
| \(r_1 \to x_2\) | reducing equivalents from TCA | NAD\(^+\)\(\to\)NADH in TCA steps |
| \(r_2 \to x_3\) | electron/proton pumping drive | NADH oxidation in chain (Q↔q, \(k_{15}\)) building \(\Delta\Psi\) |
| \(r_3 \to x_4\) | ATP synthesis drive | use of \(\Delta\Psi\) for ADP\(\to\)ATP |
| \(r_4 \to x_1\) | ATP demand / ADP feedback | ATP\(\to\)ADP demand stimulating acetyl-CoA entry/TCA |

Ring: \(x_1 \to r_1 \to x_2 \to r_2 \to x_3 \to r_3 \to x_4 \to r_4 \to x_1\).

### A.2 Equations (N=4 ADR ring)

For \(i=1..4\) (mod 4):
\[
\dot x_i = v_i,
\]
\[
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \kappa (x_{i+1}+x_{i-1}-2x_i) + \eta\, r_{i-1},
\]
\[
\dot r_i = -\mu_i r_i + \sigma_i x_i^2.
\]
No external forcing \(F\) assumed.

### A.3 Canonical “edge-of-chaos” parameters

Per-site Duffing:

* \(a = (1.0, 1.1, 0.9, 1.0)\)
* \(\gamma = (0.25, 0.20, 0.15, 0.18)\)  (TCA slowest; NADH/ΔΨ fastest)

Ring couplings:

* \(\kappa = 0.40\) (neighbour mixing)
* \(\eta = 0.60\) (resource drive)

Resources:

* \(\sigma = (0.55, 0.50, 0.60, 0.45)\)
* \(\mu_i = 4 \alpha_4^*\) with \(\alpha_4^* = 0.016\) at reference (so \(\mu_i \approx 0.064\)), scaled linearly with \(\alpha_4\) when scanning.

Effective gain at reference:
\[
\bar\sigma \approx 0.525,\quad \bar\mu \approx 0.064,\quad
\rho_{\text{ADR}} \sim \frac{\eta \bar\sigma}{\bar\mu} \approx 4.9.
\]
Interpret \(\rho_{\text{ADR}} \approx 5\) as a canonical “Krebs edge-of-chaos” gain; raising \(\mu\) (higher \(\alpha_4\)) damps to fixed points; lowering \(\mu\) yields limit cycles → period-doubling → chaos with projections akin to Grytsay’s NADH/ATP/ψ plots.

### A.4 Control parameter mapping (\(\alpha_4\) ↔ ADR leak)

* In Grytsay, \(\alpha_4\) = ATP dissipation / proton leak.
* In ADR, map \(\alpha_4\) to a shared leak: \(\mu_i(\alpha_4) = \mu_{\text{ref}} \frac{\alpha_4}{\alpha_4^*}\).
* Sweep \(\alpha_4\) (via \(\mu_i\)) to reproduce steady → oscillatory → Feigenbaum cascade → strange attractor; calibrate \(\alpha_4^*\) so the first period-doubling matches observed onset.

### A.5 Usage notes

* Use instrumented ADR scaffold (N=4, F=0, plasticity off) with the above parameters.
* Scan \(\mu\) (or \(\alpha_4\)) to build bifurcation diagrams from \(x_2, x_3, x_4\) (NADH, ΔΨ, ATP surrogates) and compute Lyapunov/CLC knees.
* Treat this as the **canonical ADR–Krebs preset**; clone the architecture and retune \((a_i,\gamma_i,\sigma_i,\mu_i)\) for other pathways (hemostasis, glycolysis, etc.) while keeping the same ring/API.

---

## Appendix B — ADR–Krebs v1 module (InstrumentedADR wrapper)

Concrete code skeleton for a drop‑in ADR–Krebs tile atop the `InstrumentedADR` scaffold (X04). Four sites, mapped to metabolic coordinates, with a single control knob `alpha` ≈ ATP dissipation / energy demand; includes helpers for trajectories and bifurcation scans.

```python
"""
adr_krebs.py — canonical 4-site Autocatalytic Duffing Ring (ADR–Krebs v1)

Sites (0-based):
  x[0] ↔ TCA throughput
  x[1] ↔ NADH / NAD+
  x[2] ↔ ΔΨ (membrane potential)
  x[3] ↔ ATP / ADP

Resources:
  r0 → NADH, r1 → ΔΨ, r2 → ATP, r3 → TCA (closes ring)

Control:
  alpha ~ energy demand / ATP dissipation (ρ_met / α4 analogue)
  eta_eff = alpha * eta_base
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict
import numpy as np
from instrumented_adr import InstrumentedADR, compute_clc_proxy

SITE_NAMES = ["TCA_flux", "NADH_ratio", "DeltaPsi", "ATP_ratio"]

@dataclass
class KrebsADRConfig:
    dt: float = 0.01
    a_TCA: float = 1.10
    a_NADH: float = 1.00
    a_dPsi: float = 1.20
    a_ATP: float = 0.90
    gamma_TCA: float = 0.22
    gamma_NADH: float = 0.20
    gamma_dPsi: float = 0.18
    gamma_ATP: float = 0.16
    mu: float = 0.15
    sigma: float = 0.60
    eta_base: float = 0.50
    F: float = 0.0
    omega: float = 1.0
    k_init: float = 0.40
    k_min: float = 0.0
    k_max: float = 1.5
    plasticity_enabled: bool = False
    eta_plast: float = 1e-3
    R_target: float = 0.6
    plasticity_every: int = 10
    neighborhood_radius: int = 1
    seed: Optional[int] = None

def make_krebs_adr(alpha: float = 1.0,
                   cfg: Optional[KrebsADRConfig] = None) -> InstrumentedADR:
    cfg = cfg or KrebsADRConfig()
    a = np.array([cfg.a_TCA, cfg.a_NADH, cfg.a_dPsi, cfg.a_ATP], dtype=float)
    gamma = np.array([cfg.gamma_TCA, cfg.gamma_NADH, cfg.gamma_dPsi, cfg.gamma_ATP], dtype=float)
    adr = InstrumentedADR(
        N=4, dt=cfg.dt,
        a=a, gamma=gamma,
        mu=cfg.mu, sigma=cfg.sigma,
        eta=cfg.eta_base * alpha,  # control knob
        F=cfg.F, omega=cfg.omega,
        k_init=cfg.k_init, k_min=cfg.k_min, k_max=cfg.k_max,
        plasticity_enabled=cfg.plasticity_enabled,
        eta_plast=cfg.eta_plast, R_target=cfg.R_target,
        plasticity_every=cfg.plasticity_every,
        neighborhood_radius=cfg.neighborhood_radius,
        seed=cfg.seed,
    )
    adr.site_names = SITE_NAMES
    adr.meta = {
        "model": "ADR-Krebs-v1",
        "mapping": {
            "x[0]": "TCA throughput",
            "x[1]": "NADH / NAD+",
            "x[2]": "DeltaPsi",
            "x[3]": "ATP / ADP",
        },
        "control_parameter": "alpha (eta gain ~ energy demand)",
    }
    return adr

def run_krebs_trajectory(alpha: float,
                         n_steps: int = 20000,
                         log_every: int = 10,
                         cfg: Optional[KrebsADRConfig] = None):
    adr = make_krebs_adr(alpha=alpha, cfg=cfg)
    log = adr.run(n_steps=n_steps, log_every=log_every)
    S_node, S_ring, diag = compute_clc_proxy(log, dt=cfg.dt if cfg else 0.01)
    return adr, log, S_ring, diag

def krebs_bifurcation_scan(alpha_values: Sequence[float],
                           n_steps_transient: int = 20000,
                           n_steps_sample: int = 20000,
                           sample_every: int = 10,
                           cfg: Optional[KrebsADRConfig] = None,
                           sample_site: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    cfg = cfg or KrebsADRConfig()
    all_alphas, x_samples = [], []
    for alpha in alpha_values:
        adr = make_krebs_adr(alpha=alpha, cfg=cfg)
        _ = adr.run(n_steps=n_steps_transient, log_every=max(1, n_steps_transient))
        log = adr.run(n_steps=n_steps_sample, log_every=sample_every)
        X = np.asarray(log["x"])
        all_alphas.extend([alpha] * len(X))
        x_samples.extend(X[:, sample_site])
    return np.asarray(all_alphas), np.asarray(x_samples)
```

Use `alpha` as the bifurcation slider. `run_krebs_trajectory` returns a log plus CLC proxy; `krebs_bifurcation_scan` builds a Feigenbaum-style scatter (alpha vs `x_TCA`). Register this module as a `BaseDynamics` in SACP/ABCP to expose ADR–Krebs as a first-class attractor model alongside Lorenz/Rössler.

---

## Appendix C — ADR–Krebs Tile (full narrative chapter)

This section captures the “Option A” narrative: how Grytsay-style metabolic chaos, MAES‑ST, and ADRs fuse into a canonical 4-site metabolic tile.

# Chapter X — The ADR–Krebs Tile: A Canonical Autocatalytic Ring for Metabolism

## 0. Why we needed a canonical metabolic tile

Grytsay and others showed that even a *single* metabolic module (Krebs + oxidative phosphorylation) can walk through the whole menagerie of dynamical regimes: fixed points, oscillations, period‑doubling cascades, and strange attractors as you tune a few control parameters, especially the ATP dissipation rate (often written as \(\alpha_4\)).

In parallel, the MAES / MAES‑ST work reframed such systems as **minimal cognitive engines**:

* Agent = the metabolic core (Krebs + ETC),
* Environment = cellular context (substrates, loads, ATP demand),
* Markov blanket = transporters + regulatory signals,
* Dynamics = an attractor bundle over a space of contexts (e.g. different \(\alpha_4\)), with horizontal, vertical, and strange‑attractor morphisms.

We’ve also built **Autocatalytic Duffing Rings (ADRs)**: rings of Duffing‑like elements \(x_i\) with local resource loops \(r_i\) that pump energy and information around a cycle, self‑tune to the edge of chaos, and can be instrumented with Lyapunov and CLC (Cognitive Light Cone) metrics.

This chapter fuses those strands into a single object:

> **The ADR–Krebs Tile** — a 4‑site Autocatalytic Duffing Ring whose states coarsely represent TCA throughput, NADH/NAD\(^+\), mitochondrial membrane potential \(\Delta\Psi\), and ATP/ADP. Its bifurcation structure in a single “leak/drive” parameter mirrors Grytsay’s Krebs diagrams, but in a compact, reusable form.

The point is *not* biochemical realism. The point is a **canonical dynamical tile** you can plug into:

* SACP / ABCP simulations,
* MAES‑ST multi‑realm models (metabolism + bioelectricity + GRNs),
* tissue‑scale ADR fields.

Once this tile is defined, “metabolism” becomes one more shaped attractor you can slot into the larger architecture.

---

## 1. From Grytsay’s Krebs model to four coarse variables

### 1.1 What Grytsay actually did (very compressed)

Grytsay’s Krebs + ETC models are fairly large ODE systems (dozens of variables), but the key story can be told in a reduced state:

* **TCA throughput**: how fast carbon is processed through the cycle.
* **Redox state**: NADH/NAD\(^+\) ratio.
* **Membrane potential \(\Delta\Psi\)**: the protonmotive force across the inner mitochondrial membrane.
* **ATP/ADP**: energy charge and its dissipation rate (\(\alpha_4\)).

As you increase ATP consumption (or equivalently, change \(\alpha_4\)), the system:

1. sits at a steady state (quiet metabolism),
2. undergoes a Hopf bifurcation to metabolic oscillations,
3. passes through period‑doublings to chaos (Feigenbaum‑style),
4. eventually loses coherent structure if driven too far.

This already *looks* like a Duffing story: double‑well energy landscape (+ drive, + damping) with parameter‑driven routes to chaos.

### 1.2 Canonical ADR–Krebs mapping

We now make that correspondence explicit. Our 4‑site ring will carry:

* \(x_1 \leftrightarrow\) **TCA throughput** (net cycle flux / carbon processing rate),
* \(x_2 \leftrightarrow\) **NADH / NAD\(^+\)** redox balance,
* \(x_3 \leftrightarrow\) **\(\Delta\Psi\)** (mitochondrial membrane potential),
* \(x_4 \leftrightarrow\) **ATP / ADP** energy charge.

Each \(x_i\) is a **Duffing‑like coordinate** in a local double‑well potential:

* One well ≈ “healthy” operating mode (e.g. balanced redox, moderate \(\Delta\Psi\), adequate ATP),
* The other ≈ “pathological” or stressed mode (e.g. chronically reduced NADH pool, collapsed \(\Delta\Psi\), ATP depletion).

We think of the 4‑tuple \((x_1,\dots,x_4)\) as a *coarse phase‑space chart* of the full enzymatic network, with an ADR ring providing a generic route to rich dynamics.

---

## 2. The ADR–Krebs ring equations

We start from the standard ADR form:

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i(x_i)
= \kappa (x_{i+1} + x_{i-1} - 2 x_i) + \eta r_{i-1} + u_i(t),
\]
\[
\dot r_i = -\mu r_i + \sigma x_i^2.
\]

Indices \(i\) are modulo 4 (a ring). Here:

* \(x_i\) = metabolic coarse variable \(i\),
* \(V_i(x)\) = local potential (typically double‑well),
* \(\gamma_i\) = local damping,
* \(\kappa\) = diffusive coupling along the ring (fast bidirectional influence),
* \(r_i\) = resource / catalytic pool “produced” by site \(i\),
* \(\eta\) = feed‑forward resource coupling (autocatalytic drive \(r_{i-1} \to x_i\)),
* \(\mu\) = resource leak / degradation,
* \(\sigma\) = resource production from local activity,
* \(u_i(t)\) = external inputs (e.g. substrate supply, bioelectric modulation, etc.).

### 2.1 Choosing potentials: double wells as metabolic regimes

A simple canonical choice:

\[
V_i(x) = \tfrac14 x^4 - \tfrac{a_i}{2}x^2,
\]

so that

\[
\partial_{x_i} V_i = x_i^3 - a_i x_i.
\]

* The two wells at \(x = \pm \sqrt{a_i}\) represent two operating regimes.
* We can bias them later (add a cubic term \(c_i x^3\)) if we want one mode to be preferred.

For now, think:

* \(x_1\) healthy well: “robust net flux”; other well: “stalled / anaplerotic bias”.
* \(x_2\) healthy well: balanced NADH/NAD\(^+\); other: chronically reduced or oxidised pool.
* \(x_3\) wells: high vs. low \(\Delta\Psi\).
* \(x_4\) wells: high vs. low ATP/ADP.

### 2.2 Full ADR–Krebs equations (dimensionless form)

Write each site in first‑order form with velocity \(v_i = \dot x_i\):

\[
\dot x_i = v_i,
\]
\[
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3
\;+\; \kappa (x_{i+1} + x_{i-1} - 2x_i)
\;+\; \eta r_{i-1} + u_i(t),
  \]
  \[
  \dot r_i = -\mu r_i + \sigma x_i^2.
  \]

Map these to biology as:

* **Site 1 (TCA flux \(x_1\))**

  * \(u_1(t)\): acetyl‑CoA supply / substrate inflow (e.g., glycolytic + fatty acid inputs).
  * \(r_4\) (previous in ring) influences \(x_1\): we interpret \(r_4\) as ATP‑driven “demand”, closing a feedback loop from energy use back to throughput.

* **Site 2 (NADH/NAD\(^+\) \(x_2\))**

  * Coupled to \(x_1\) (flux) and \(x_3\) (\(\Delta\Psi\)): flux generates NADH; membrane potential influences NADH oxidation in ETC via \(r_1\).

* **Site 3 (\(\Delta\Psi\) \(x_3\))**

  * Driven by redox \(x_2\) (NADH feeds ETC) and ATP synthase load \(x_4\) via \(r_2\).

* **Site 4 (ATP/ADP \(x_4\))**

  * Driven by \(\Delta\Psi\) and ATP demand (external input \(u_4(t)\)); its activity produces the resource \(r_4\) that feeds back to throughput at site 1.

This gives a **closed autocatalytic ring**: throughput → redox → \(\Delta\Psi\) → ATP → back to throughput, with each site implemented as a Duffing‑like unit.

---

## 3. Matching Grytsay’s bifurcation parameter

We now introduce a single “master knob” that plays the role of Grytsay’s \(\alpha_4\) — a global energy dissipation / load parameter.

### 3.1 Defining the control parameter \(\mu_0\)

We choose \(\mu_0\) to modulate:

* The resource leak \(\mu\) (how fast catalytic support decays),
* Possibly the damping \(\gamma_i\) (how strongly the system resists change),
* And/or a global ATP demand term in \(u_4(t)\).

A simple canonical choice:

\[
\mu = \mu_0,
\qquad
u_4(t) = -\beta_{\text{ATP}} \mu_0,
\]

so increasing \(\mu_0\):

* drains resources faster (harder to sustain autocatalytic cycling),
* increases ATP draw (higher load).

By sweeping \(\mu_0\) over a range \([\mu_{\min}, \mu_{\max}]\), we can reproduce the classical scenario:

* \(\mu_0\) small: sufficient resource, manageable load → stable fixed point or small oscillation;
* \(\mu_0\) moderate: Hopf to limit cycles, then period‑doubling cascade;
* \(\mu_0\) larger still: strange attractor in the \((x_1,x_3)\) and \((x_2,x_4)\) planes,
  reminiscent of Grytsay’s strange attractors in TCA/ETC variables.

### 3.2 Canonical parameter ranges

For a canonical “textbook tile”, we can pick dimensionless parameters like:

* \(a_i \in [0.8, 1.2]\) (slight heterogeneity),
* \(\gamma_i \in [0.15, 0.25]\),
* \(\kappa \sim 0.2{-}0.4\),
* \(\sigma \sim 0.5\),
* \(\eta \sim 0.5{-}1.0\),
* \(\mu_0\) as the main scanned parameter (e.g. \(0.01 \to 0.08\)).

The exact numbers are *tunable*; what matters is that:

* for low \(\mu_0\) the ring is under‑damped and resource‑rich enough to oscillate,
* for intermediate \(\mu_0\) you get mixed periodic / chaotic regimes,
+* for high \(\mu_0\) the ring collapses into a low‑energy attractor (e.g. ATP‑starved basin).

You now have a **metabolic bifurcation diagram in one slider**.

---

## 4. Instrumenting the ADR–Krebs tile

The rest of the ADR ecosystem gives us powerful tooling to *see* what this tile is doing.

### 4.1 Lyapunov & chaos diagnostics

Because the ADR–Krebs tile is an ADR, we can:

* Estimate the largest Lyapunov exponent \(\lambda_{\max}(\mu_0)\) via twin‑trajectory separation,
* Construct bifurcation diagrams by plotting Poincaré‑section samples of, say, \(x_4\) vs. \(\mu_0\),
* Visualise phase portraits in \((x_1,x_3)\), \((x_2,x_4)\) to compare qualitatively with Grytsay plots.

In practice, we can reuse the **instrumented ADR** scaffold (iADR) from the simulation appendix:

* Treat \(N=4\) instead of 16,
* Label \(x_i\) with biological meaning,
* Interpret the existing CLC and coherence metrics as “metabolic cognitive capacity” instead of generic ring cognition.

### 4.2 Cognitive Light Cone (CLC) as “metabolic intelligence”

For each site \(i\) (TCA, NADH, \(\Delta\Psi\), ATP), we can:

* Compute a local coherence time (how long the variable remains in a dynamical regime without collapsing),
* Compute an autocorrelation horizon (how far into past/future its state is predictive),
* Compute a spatial radius (how strongly it influences the other three nodes across the ring).

This gives a per‑site CLC score:

\[
S_{\text{CLC},i}
\approx C_{\infty,i}\sqrt{\tau_{\text{past},i}\tau_{\text{future},i}} R_{\text{CLC},i},
\]

and a ring‑level metabolic CLC:

\[
S_{\text{CLC, Krebs}} = \frac{1}{4}\sum_{i=1}^4 S_{\text{CLC},i}.
\]

Then \(\mu_0\)-sweeps give a **metabolic cognitive knee**:

* Subcritical regime: low CLC (system is too rigid / laminar),
* Edge‑of‑chaos band: high CLC (rich, structured variability, strong internal coordination),
* Over‑driven: CLC collapses (metabolism becomes too noisy to sustain coherent function).

That story fits naturally with the “homeodynamics” and minimal‑cognitive‑engine view in the MAES–Krebs paper.

---

## 5. Plugging the ADR–Krebs tile into SACP / MAES‑ST

Now we treat this tile as just another **SACP module** in the SACP suite / ABCP architecture:

### 5.1 SACP interface

For the ADR–Krebs ring:

* **Sense:** expose \((x_1,\dots,x_4)\), plus composite observables:

  * TCA flux index \(F_{\text{TCA}} \sim x_1\),
  * Redox index \(R_{\text{NADH}} \sim x_2\),
  * Membrane potential index \(\Psi \sim x_3\),
  * Energy charge index \(E_{\text{ATP}} \sim x_4\),
  * Chaos / CLC diagnostics.

* **Act:** allow external controllers to inject into \(u_i(t)\):

  * Substrate pulsing (into \(u_1\)),
  * Redox perturbations (into \(u_2\)),
  * ETC uncouplers / channel modulation (into \(u_3\)),
  * ATP demand profiles (into \(u_4\)).

* **Couple:** use the ring’s \(\kappa, \eta, \mu_0\) as knobs to couple it to:

  * Bioelectric fields (via \(\Delta\Psi\) ↔ ADR field tiles),
  * GRN tiles (transcriptional control of resource parameters),
  * Tissue mechanics (ATP-dependent contractility).

* **Perturb:** run shocks, parameter jumps, and structural changes as MAES vertical morphisms (switching contexts in the MAES‑ST sheaf; e.g., hypoxia, uncoupling, developmental transitions).

### 5.2 Cross‑realm coupling (metabolism ↔ bioelectricity & GRNs)

Using MAES‑ST’s cross‑realm coupling, we can:

* Let a **bioelectric ADR field** modulate \(\mu_0\) or \(\eta\) (i.e. local voltage gates metabolic chaos),
* Let a **GRN tile** modulate \(a_i, \gamma_i\) (gene expression changes the depth and symmetry of the metabolic wells),
* Let metabolic CLC feed back to GRN / bioelectric modules as a contextual signal (e.g., stress response).

In sheaf language, each context \(U \subset B\) (e.g. “normoxia”, “hypoxia”, “high load”) gets a different ADR–Krebs parameter set \(M(U)\), with vertical morphisms capturing transitions between them.

---

## 6. Why this tile is reusable

The magic of the ADR–Krebs tile is that it’s **not special to Krebs**:

* Any autocatalytic chemical cycle with a few key coarse variables (flux, energy, redox, some “field”) can be compressed into a small ADR ring with double‑well local dynamics plus resource loops.
* The same architecture applies to:

  * Glycolysis (flux, NADH, PFK “field”, ATP),
  * Hemostasis (thrombin, fibrin, prostanoid balance, platelet activation),
  * Generic enzymatic feedback loops exhibiting oscillations / chaos.

You get:

* a *canonical shape* of the attractor (Duffing‑ring strange attractor),
* a *canonical set of knobs* (\(\mu_0, \kappa, \eta, a_i, \gamma_i\)),
* and a *canonical instrumentation* (Lyapunov, CLC, SACP hooks).

The biological semantics change with the \(x_i \leftrightarrow\) mapping; the math and the software tile stay the same.

---

## 7. Next chapter: ADR–GRN and beyond

In this chapter we’ve:

* compressed a Grytsay‑style Krebs model into a 4‑site ADR ring,
* defined a mapping \(x_1\ldots x_4 \leftrightarrow (\text{flux},\text{NADH},\Delta\Psi,\text{ATP})\),
* introduced a master leak/drive parameter \(\mu_0\) mimicking \(\alpha_4\),
* and plugged the tile into the SACP / MAES‑ST architecture.

The natural “other option” is to do the same for **Gene Regulatory Networks** (GRNs) — an ADR–GRN tile where sites represent mRNA, protein, chromatin accessibility, and cellular state, again with a single bifurcation parameter acting like Grytsay’s \(\alpha_4\) but for transcriptional energy status.

That can be written as a companion chapter; the structure will be almost identical, but with different biological labels and a slightly different pattern of couplings.

For now, the ADR–Krebs tile gives us a canonical, executable metabolic block: a little four‑node ring you can drop into any model that needs “a metabolically alive piece of tissue”, complete with its own knees, chaos, and cognitive light cone.

---

## Appendix D — ADR–Bioelectric Tile: Steering Vmem Attractors

A companion tile to ADR–Krebs: same 4-site ADR architecture, but with bioelectric/morphogenetic semantics.

### 0. Motivation

Bioelectric tissues exhibit local bistability (depolarised vs hyperpolarised), field-level attractors (default anatomical patterns), and can get stuck in pathological basins (tumor-like depolarised islands, mispatterned setpoints). Clinically, we want handles to push a patch from a negative attractor back to a healthy one. An ADR ring is a natural substrate: bistable sites, autocatalytic loop, and self-tuning toward edge-of-chaos.

### 1. Coarse variables (ring sites)

1. \(x_1\): local membrane potential (Vmem).
2. \(x_2\): fast ionic drive / channel state (aggregate excitability/leak).
3. \(x_3\): connectivity / gap-junction openness (coupling to neighbours).
4. \(x_4\): slow morphogenetic setpoint / bioelectric memory (target Vmem & coupling).

Each \(x_i\) has a double well: healthy vs negative regime.

### 2. ADR–Bioelectric equations

Reuse ADR form (indices mod 4):
\[
\dot x_i = v_i,
\]
\[
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \kappa (x_{i+1}+x_{i-1}-2x_i) + \eta r_{i-1} + u_i(t),
\]
\[
\dot r_i = -\mu r_i + \sigma x_i^2.
\]

Site semantics:

- Site 1 (Vmem): \(r_4\to x_1\) encodes setpoint pull; \(u_1\) = electrodes/fields/optogenetics.
- Site 2 (fast channels): \(r_1\to x_2\) activity-dependent channel modulation; \(u_2\) = channel drugs/opsins.
- Site 3 (connectivity): \(r_2\to x_3\) Ca/phospho-dependent GJ state; \(u_3\) = GJ modulators/mechanical cues.
- Site 4 (setpoint): \(r_3\to x_4\) long-lived connectivity → transcriptional programs; \(u_4\) = epigenetic/developmental cues.

Ring closure \(r_4 \to x_1\): setpoint → Vmem → channels → connectivity → setpoint (autocatalytic loop).

### 3. Healthy vs negative attractors

Double wells per site yield a healthy attractor (all in healthy wells) and negative attractors (e.g., depolarised/isolation/repatterned setpoint). Resource coupling makes whole-pattern basins.

### 4. Master bioelectric load/leak parameter

Define \(\ell_0\) to increase leak and load:
\[
\mu = \mu_0 + c_\mu \ell_0,\quad \gamma_i = \gamma_{i,0} + c_{\gamma,i}\ell_0,\quad u_1 = u_{1,0} - \beta_{\text{load}}\ell_0.
\]
Ramping \(\ell_0\): low → healthy attractor stable; mid → oscillatory/chaotic (good for rewrites); high → flattened/unpatterned. Protocol to escape bad basins: raise \(\ell_0\) (enter chaos), bias toward healthy, lower \(\ell_0\).

### 5. Control handles

- \(u_1\): direct Vmem drive (electrodes, optogenetics).
- \(u_2\), \(a_2,\gamma_2\): channel modulation (drugs, opsins).
- \(u_3\), \(a_3\): gap-junction modulation (connexins, stretch).
- \(u_4\), \(a_4\): setpoint rewrites (epigenetic/developmental cues).

### 6. Crossing basins

- **Direct flip:** Vmem pulse + temporary tilt of \(a_1\), let resources pull others, remove bias.
- **Chaos-assisted reset:** raise \(\ell_0\) → chaos; bias \(a_4,u_4\) so healthy setpoint is sole deep well; lower \(\ell_0\); lightly increase damping to stabilise.

### 7. Self-tuning via bioelectric free-energy

Define local functional for tile \(j\):
\[
G_j = w_V (x_{1,j}-x^\star_{1,j})^2 + w_{\text{shape}}\sum_{k=2}^4 (x_{k,j}-x^\star_{k,j})^2 + w_\lambda (\lambda_j - \lambda_{\text{band}})^2.
\]
Slow parameters follow \(\dot\theta_j = -\varepsilon \partial_{\theta_j} G_j\), keeping Vmem near target, internal state aligned, and chaos in-band. Sum over tiles for tissue-level homeodynamics.

### 8. From tiles to bioelectric ADR field

Continuum ADR field: \(\partial_t^2 \phi + \gamma \partial_t \phi = D\nabla^2\phi - \alpha \phi - \beta \phi^3 + J + \eta \rho\), \(\partial_t \rho = -\mu \rho + \sigma g(\phi)\). Tiles homogenise to \(\phi\) (Vmem) and \(\rho\) (resource). Wound = local kick; healing = return to desired attractor \(\phi_+\); eliminate negative attractor \(\phi_-\) via parameter shaping + stimulation.

### 9. Scenarios

- **Depolarised/tumor-like island:** raise \(\ell_0\) (weaken bad attractor), pulse \(u_1,u_3\) toward healthy wells, tilt \(a_4/u_4\) to rewrite setpoint, let self-tuning minimise \(G_j\), lower \(\ell_0\), modestly raise damping.
- **Wound healing:** boost coupling \(u_3\) at wound edge, keep wound tiles slightly more chaotic, let \(G_j\) drive setpoint/Vmem toward neighbours, then restore baseline parameters.

### 10. MAES‑ST view

Each tile = agent; environment = surrounding tissue; blanket = boundary (GJs, extracellular space). Vertical morphisms = slow parameter flow \(\dot\theta = -\partial_\theta G\); \(\eta_{\text{SA}}\) morphisms = chaos episodes under high \(\ell_0\); horizontal = ordinary Vmem dynamics. The bioelectric ADR field is a lattice of micro-agents maintaining a target Vmem pattern and escapable from bad basins via structured interventions.
```
