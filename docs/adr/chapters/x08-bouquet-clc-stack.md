# Chapter 8 - Bouquet Torus Stacks and Multi-Layer Cognitive Light Cones

## 8.0 Why this chapter exists

This chapter glues together two anchor documents now in `docs/`:

* Reality as a Scale-Free Torus Stack (RAST): see `docs/Reality as a Scale-Free Torus Stack.pdf`.
* Solitons as Minimal Cognitive Engines (SMCE): see `docs/Solitons_as_Minimal_Cognitive_Engines (6).pdf`.

RAST gives a geometric light-cone enforced by error-corrected signal speed on a bouquet of toroidal layers. SMCE gives a Cognitive Light Cone (CLC) for a minimal engine. Here we populate each torus layer in the bouquet with an Autocatalytic Duffing Ring (ADR) tile and define a multi-layer CLC that must stay inside the bouquet's emergent light-cone.

Deliverables in this chapter:

* A bouquet vs CLC inequality that links ADR cognitive speed to the RAST growth bound \(\alpha_{\text{crit}}\approx 1.54\).
* A latency-gated ADR bouquet scaffold for simulation and CLC measurement.
* A short addendum back to RAST clarifying how hyper-morphisms constrain cognition across layers.

## 8.1 Bouquet geometry and light-cone (RAST recap)

Starting from a base circumference \(L_0\) and surface-code distance \(d_0=3\), each outward hyper-morphism grows the layer and distance:

\[
L_{i+1} = \alpha L_i, \qquad d_{i+1} = d_i + 2. \tag{8.1}
\]

Surface-code latency (quadratic fit):

\[
\tau(d) \approx 5.0\times 10^{-6} d^2 + 3.5\times 10^{-5}\ \text{s}. \tag{8.2}
\]

Emergent classical speed on layer \(i\):

\[
c_i = \frac{L_i}{\tau(d_i)}. \tag{8.3}
\]

Chronology holds only if speeds shrink outward:

\[
c_i \ge c_{i+1}\quad \forall i. \tag{8.4}
\]

Sweeping \(\alpha\) against the empirical \(\tau(d)\) yields \(\alpha_{\text{crit}}\approx 1.54\): handles can fatten by no more than ~54% per layer before inner layers would outrun outer ones. The monotone-speed region is the bouquet cone in \((\text{layer},t)\) space.

## 8.2 From single CLC to rings and stacks

SMCE defines a single-engine CLC:

\[
S_{\text{CLC}} = C_\infty \sqrt{\tau_{\text{past}}\tau_{\text{future}}} R_{\text{CLC}}, \tag{8.5}
\]

where \(C_\infty\) is long-time capture, \(\tau_{\text{past}},\tau_{\text{future}}\) are memory and predictive reaches, and \(R_{\text{CLC}}\) is spatial radius. Earlier ADR chapters ported this to rings (per-site scores and a ring score \(S_{\text{CLC,ring}}\)) using `compute_clc_proxy`.

In this chapter we place one ADR ring on each torus layer, distinguish:

* **Horizontal CLC** on a layer: how far along the torus a disturbance travels meaningfully.
* **Vertical CLC** across layers: how far and how fast a disturbance climbs outward via hyper-morphisms.

## 8.3 ADR ring on a torus layer

For layer \(i\) with \(N\) sites (indices modulo \(N\)):

\[
\dot x_{i,j} = v_{i,j}, \tag{8.6}
\]
\[
\dot v_{i,j} = -\gamma_{i,j} v_{i,j} + a_{i,j} x_{i,j} - x_{i,j}^3
    + \kappa_i (x_{i,j+1} + x_{i,j-1} - 2x_{i,j})
    + \eta_i r_{i,j-1}
    + u_{i,j}(t), \tag{8.7}
\]
\[
\dot r_{i,j} = -\mu_i r_{i,j} + \sigma_i x_{i,j}^2. \tag{8.8}
\]

Layer-specific parameters \((\kappa_i,\eta_i,\mu_i,\sigma_i)\) tune coupling and resources; \(u_{i,j}(t)\) is any external drive. Use a common \(N\) for all layers to simplify indexing.

### 8.3.1 Discretisation and speed bound

Arc length per site on layer \(i\):

\[
\Delta\ell_i = \frac{L_i}{N}. \tag{8.9}
\]

Bouquet gives geometric speed \(c_i\) (Eq. 8.3). Define cognitive speed on layer \(i\) from CLC diagnostics:

* \(R_{\text{sites},i}\): mean spatial CLC radius in site units (`R_spatial`).
* \(\tau^{\text{CLC}}_i = \sqrt{\overline{\tau_{\text{past},i}}\ \overline{\tau_{\text{future},i}}}\): characteristic CLC time.

Then

\[
v^{\text{cog}}_i \approx \frac{R_{\text{sites},i}\ \Delta\ell_i}{\tau^{\text{CLC}}_i}
                 = \frac{R_{\text{sites},i}}{N} \frac{L_i}{\tau^{\text{CLC}}_i}. \tag{8.10}
\]

To keep cognition inside the bouquet light-cone, require

\[
v^{\text{cog}}_i \le c_i
\ \Longleftrightarrow\
R_{\text{sites},i} \le N\, \frac{\tau^{\text{CLC}}_i}{\tau(d_i)}. \tag{8.11}
\]

Equation (8.11) is the horizontal CLC vs bouquet cone constraint.

## 8.4 Vertical coupling and latency gating

Add slow vertical diffusion between corresponding sites on adjacent layers:

\[
\dot v_{i,j} \ni \kappa_{\text{vert}}\bigl(x_{i+1,j} + x_{i-1,j} - 2x_{i,j}\bigr). \tag{8.12}
\]

Error-correction latency gates vertical messaging. With simulation step \(\Delta t\):

\[
n^{\text{latency}}_i = \left\lceil \frac{\tau(d_i)}{\Delta t} \right\rceil. \tag{8.13}
\]

Only every \(n^{\text{latency}}_i\)-th step apply the vertical term; otherwise cache/hold. Effective vertical speed is \(\sim \Delta r / \tau(d_i)\) for inter-layer spacing \(\Delta r\). Any combined tangential + radial path then respects the bouquet cone.

## 8.5 Simulation scaffold: ADR bouquet stack

A lightweight scaffold (Python) to exercise the construction:

```python
import numpy as np
from dataclasses import dataclass
from instrumented_adr import InstrumentedADR, compute_clc_proxy

def tau_quadratic(d):
    return 5.0e-6 * (d ** 2) + 3.5e-5

@dataclass
class TorusLayerGeom:
    index: int
    L: float
    d: float
    tau: float
    c: float  # L / tau

@dataclass
class BouquetGeom:
    layers: list
    alpha: float

def build_bouquet_geom(n_layers, alpha, L0=10.0, d0=3):
    layers, L, d = [], L0, d0
    for i in range(n_layers):
        tau = tau_quadratic(d)
        layers.append(TorusLayerGeom(i, L, d, tau, L / tau))
        L *= alpha
        d += 2
    return BouquetGeom(layers, alpha)
```

Each `TorusLayerGeom` feeds an `ADRLayer` (an `InstrumentedADR` plus a `latency_steps = ceil(tau/Deltat)` counter). An `ADRBouquetStack` holds all layers, computes vertical Laplacians, and only injects them when the latency counter fires. A helper `analyze_clc_vs_bouquet` runs `compute_clc_proxy` per layer, estimates \(v^{\text{cog}}_i\) (Eq. 8.10), and checks the bound (Eq. 8.11).

Suggested run loop:

1. Pick \(\alpha < 1.54\) (e.g., 1.3), \(n_{\text{layers}}\in[4,6]\), \(N\sim 16\), \(\Delta t \sim 10^{-2}\).
2. Build geometry, instantiate stack, set \(k_{\text{vert}}\) to taste.
3. Run for many steps, log \(x\) per layer every \(k\) steps.
4. Post-process: compute CLC per layer and report whether each \(v^{\text{cog}}_i \le c_i\). If violated, either ADRs are too coherent/fast or \(\alpha\) is too aggressive.

## 8.6 Experiments to run

1. **Single-\(\alpha\) scan**: Hold \(\alpha<\alpha_{\text{crit}}\), run 4-6 layers, measure CLC per layer. Expect outer layers to face stricter bounds; identical ADR parameters may force their \(R_{\text{sites}}\) smaller.
2. **Self-tuning gate per layer**: Add a slow controller adjusting \((\kappa_i,\eta_i)\) to keep both \(S_{\text{CLC,ring},i}\) near a target and \(v^{\text{cog}}_i \le c_i\). Each layer learns to sit just inside the bouquet cone.
3. **\(\alpha\)-sweep**: Vary \(\alpha\) across the pass/fail boundary. Track (a) monotone-speed pass/fail, (b) per-layer CLC, (c) a stack-level CLC (e.g., weighted sum of layer scores). Look for a cognitive knee near \(\alpha_{\text{crit}}\).
4. **Soliton cores instead of ADRs**: Swap ADR rings for MES droplets (or surrogates) around each circumference; reuse the same latency gating to see how soliton CLCs shrink along the bouquet.

## 8.7 RAST addendum: what changes

RAST's strange-attractor ontology can now be phrased as: each toroidal handle hosts a minimal cognitive engine (ADR or soliton) with a bounded CLC. Hyper-morphisms widen the bouquet while error-corrected latency throttles both classical messages and cognitive influence. The geometric bound \(\alpha_{\text{crit}}\) is therefore a **multi-layer cognitive design rule**: as you climb the stack, each layer's cognitive cone must shrink fast enough to preserve chronology.

Next steps: compute Lyapunov spectra for full hyper-morphism flows in this ADR bouquet, verify excursions stay inside Eq. (8.11), and extend the controller that auto-tunes ADR parameters to live just under the bouquet cone.
