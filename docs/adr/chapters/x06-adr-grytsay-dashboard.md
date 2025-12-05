# ADR–Grytsay Reinterpretation for the ADR Chemistry Dashboard

*Condensed chapters distilled from the ADR–Grytsay Model Reinterpretation chat; meant to slot into the Autocatalytic Duffing Ring primer.*

---

## Chapter 1 — From Metabolic Chaos to Autocatalytic Duffing Rings

### 1.1 Why we need a canonical tile

Classical biochemical models like Grytsay’s Krebs cycle and oxidative phosphorylation already showed what we care about: **deterministic chaos in real metabolic networks**, with Feigenbaum-style period-doubling cascades as you dial a single control parameter, usually an energy dissipation / ATP turnover rate such as \(\alpha_4\).

These models are rich but *bespoke*: dozens of variables and rate constants, hand-tuned for a specific pathway. They are great for papers, but terrible as **reusable building blocks** in a general “complexity accelerator” or SACP suite.

Our goal in this project is:

> Turn complicated enzymatic networks into **canonical dynamical tiles** – Autocatalytic Duffing Rings (ADRs) – that preserve the key bifurcation and attractor structure, but live in a small state space, with a standard interface, and can be dropped into anything.

This chapter takes the **Krebs cycle** as the flagship example and shows how to go from Grytsay’s chaotic metabolic model to a **4-dimensional ADR–Krebs surrogate**.

### 1.2 Grytsay’s Krebs cycle as a MAES system

In *Strange Attractors in Metabolism* we already reinterpreted the Krebs cycle using the **MAES-ST** formalism: Mirrored Agent–Environment Systems with a stochastic, topos-theoretic treatment of context.

At a given context (oxygen, substrate load, ATP demand):

* **Agent (A)**: internal metabolic state:
  * concentrations of TCA intermediates,
  * NADH/FADH\(_2\),
  * mitochondrial potential \(\Delta\Psi\),
  * local ATP/ADP levels.
* **Environment (E)**: cell-scale conditions:
  * cytosolic ATP demand,
  * nutrient and oxygen supply,
  * broader redox pools, calcium, etc.
* **Markov Blanket (B)**:
  * transporters, dehydrogenase complexes, ATP synthase – the interface where the cycle feels the environment and pushes back.
* **Interaction dynamics (I)**:
  * Grytsay’s ODEs, where a control parameter like ATP dissipation \(\alpha_4\) plays the role of **context**.

MAES-ST rephrases this as a sheaf of local systems \(M(U) = (A(U), E(U), B(U), I(U))\) over a base space of contexts \(U\) (ranges of \(\alpha_4\), oxygen, etc.), and distinguishes:

* **Horizontal morphisms (h)**: ordinary time evolution in a fixed regime.
* **Vertical morphisms (v)**: structural or parametric jumps, like moving to a new band of \(\alpha_4\).
* **Strange-attractor morphisms (\(\eta_{\mathrm{SA}}\))**: chaotic trajectories living on a strange attractor for a given context.

Experimentally and in Grytsay’s models, sweeping \(\alpha_4\) drives the system from a fixed point → limit cycles → period-doubling → chaos, with a well-defined chaotic window where the largest Lyapunov exponent \(\lambda_1 > 0\).

This is the **raw material** we want the ADR tile to capture.

### 1.3 Compressing Krebs into four canonical coordinates

To make a **portable tile**, we compress the high-dimensional metabolic model into four coarse variables:

* \(x_1 \leftrightarrow\) **TCA throughput** (effective flux around the cycle),
* \(x_2 \leftrightarrow\) **NADH/NAD\(^+\)** redox balance,
* \(x_3 \leftrightarrow\) **\(\Delta\Psi\)** (mitochondrial membrane potential),
* \(x_4 \leftrightarrow\) **ATP/ADP ratio**.

These four already appear as aggregate variables in the MAES-ST analysis of Grytsay’s models and in the “minimal cognitive engine” reading of metabolism: throughput, redox, voltage, and ATP are the main knobs that matter dynamically.

Each \(x_i\) is treated as a **Duffing-like coordinate** sitting in a double-well potential:

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i(x_i) = I_i(t),
\]

with \(V_i(x_i) \approx \tfrac14 x_i^4 - \tfrac{a_i}{2} x_i^2\) (biologically: two qualitative regimes, e.g., low vs high flux or healthy vs pathological redox).

We then couple these four sites in a **ring**, adding:

* diffusive coupling terms between neighbouring \(x_i\),
* an **autocatalytic resource loop** \(r_i\) that integrates local activity and drives the next site in the ring (Section 1.4).

The claim is not that these are literally four molecules, but that they define a **4D coarse phase space** within which we can reproduce:

* the same **route to chaos** in a single effective control parameter,
* similar attractor projections in biologically meaningful planes.

### 1.4 The ADR–Krebs ring

An Autocatalytic Duffing Ring (ADR) has:

* Duffing-like states \(x_i\),
* velocities \(\dot x_i\),
* a resource variable \(r_i\) at each site,
* ring topology (indices modulo \(N\)).

The minimal ADR equations (for \(N = 4\) as ADR–Krebs) are:

\[
\dot x_i^{(1)} = x_i^{(2)},
\]

\[
\dot x_i^{(2)} = -\gamma_i x_i^{(2)} + a_i x_i^{(1)} - \bigl(x_i^{(1)}\bigr)^3 + \kappa \bigl(x_{i+1}^{(1)} + x_{i-1}^{(1)} - 2 x_i^{(1)}\bigr) + \eta r_{i-1} + F \cos(\omega t),
\]

\[
\dot r_i = -\mu r_i + \sigma \bigl(x_i^{(1)}\bigr)^2,
\]

with indices taken mod \(4\).

Interpretation in the Krebs context:

* \(x_1\): TCA throughput; \(r_4\) roughly tracks upstream substrate supply.
* \(x_2\): NADH/NAD\(^+\); driven by \(r_1\) (flux through dehydrogenases).
* \(x_3\): \(\Delta\Psi\); driven by \(r_2\) (electron transport).
* \(x_4\): ATP/ADP ratio; driven by \(r_3\) (protonmotive force and ATP synthase).

The **resource loop** \(r_i\) implements “what happens here fuels the next step”. A burst of \(x_1\) (flux) raises \(r_1\), which drives \(x_2\) (NADH), and so on around the ring.

We then pick a **single effective control parameter**:

* for example the product \(\alpha_{\mathrm{eff}} = \sigma \eta / \mu\), or a global “ATP leak” \(\gamma_{\mathrm{ATP}}\) that appears in the damping of \(x_4\),

and sweep it in a range chosen to mimic Grytsay’s \(\alpha_4\). For appropriate values, the ADR–Krebs ring shows:

* fixed points → limit cycles → period-doubling → high-dimensional strange attractors, with a positive largest Lyapunov exponent, just like the detailed metabolic models.

The **exact numerical fit** to Grytsay’s diagrams is a calibration exercise, but the structural point is:

> A 4-site ADR ring can reproduce the same qualitative bifurcation structure and chaotic regime as a full metabolic model, using four interpretable coarse variables.

### 1.5 MAES-ST reading of ADR–Krebs

In the MAES-ST language, ADR–Krebs is simply another local object \(M(U)\):

* **Agent (A(U))**: \(x_1, \dots, x_4, r_1, \dots, r_4\).
* **Environment (E(U))**: external drive \(F\), leak parameters, substrate inputs.
* **Blanket (B(U))**: the resource variables \(r_i\) and any explicit interface fluxes.
* **Dynamics (I(U))**: the ADR ODEs at context \(U\).

Sweeping the effective \(\alpha_{\mathrm{eff}}\) is a **vertical morphism** in context space. For a given \(\alpha_{\mathrm{eff}}\), the trajectory of \((x_i, r_i)\) is a **horizontal morphism**; in the chaotic window, those trajectories are \(\eta_{\mathrm{SA}}\) strange-attractor morphisms.

This gives us exactly what we wanted:

* a **canonical 4D tile**, tightly linked to biochemical meaning;
* a clear **control parameter** to play “edge of chaos” games with;
* a standard form that we can clone to other autocatalytic chemistries:
  * haemostasis,
  * glycolysis,
  * simple gene regulatory motifs, etc.

The rest of the book treats this ADR tile as the default dynamical primitive.

---

## Chapter 2 — The Instrumented ADR Tile and Cognitive Knees

Chapter 1 explained *why* we want ADRs and roughly how ADR–Krebs is constructed. This chapter treats ADRs as a **software object** with:

* explicit equations,
* a standard interface (SACP),
* built-in instrumentation for “cognitive knees”.

### 2.1 The Autocatalytic Duffing Ring, cleanly

The generic ADR definitions (from Chapter 17 in the notes) are:

* \(N\) sites on a ring.
* Each site \(i\) has:
  * state \(x_i(t)\),
  * local potential \(V_i(x_i)\) (Duffing-like),
  * damping \(\gamma_i\),
  * resource \(r_i(t)\).

Dynamics:

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i = \kappa (x_{i+1} + x_{i-1} - 2 x_i) + \eta r_{i-1} + \xi_i(t),
\]

\[
\dot r_i = -\mu r_i + \sigma x_i^2.
\]

Key parameters:

* \(\kappa\): diffusive coupling (“spring” along the ring).
* \(\eta\): strength of resource drive from site \(i-1\).
* \(\mu, \sigma\): resource decay and production.
* \(\xi_i(t)\): optional noise.

**Autocatalysis** in this language means:

* the trivial rest state is not globally attracting,
* a local kick anywhere can start a self-sustaining circulating pattern,
* cutting the ring (breaking the feedback path) destroys that pattern.

### 2.2 Ring-level strange attractors and “ring cognition”

Even a single Duffing oscillator is chaotic in the right parameter ranges. A heterogeneous ring of them with a resource loop exhibits:

* traveling waves,
* multi-site limit cycles,
* high-dimensional strange attractors with positive Lyapunov exponents.

We denote the full state as:

\[
X = (x_1, \dot x_1, \dots, x_N, \dot x_N, r_1, \dots, r_N),
\]

and for a parameter vector \(\theta\) the flow \(\dot X = F_\theta(X)\) has an attractor \(A(\theta)\). The **Lyapunov spectrum** \(\{\lambda_k\}\) and derived fractal dimension give us a handle on “how chaotic” the ring is.

Empirically (and in toy simulations in later chapters), one sees:

* low drive → quiescent or single-well behaviour,
* intermediate drive → structured patterns (memory-like),
* high drive → “thermal chaos”.

We want to live near the **“edge of chaos”** in the middle regime.

### 2.3 Cognitive Light Cones (CLC) for ADRs

To talk about “cognition” in a ring, we introduced **Cognitive Light Cone (CLC)** metrics, first for MES solitons and then for ADRs and rings.

For each site \(i\), a proxy CLC score:

\[
S_{\text{CLC}, i} \approx C_{\infty, i} \sqrt{\tau_{\text{past}, i} \tau_{\text{future}, i}}\,R_{\text{CLC}, i},
\]

where:

* \(C_{\infty, i}\): how long site \(i\) stays in a **coherent regime** (not frozen, not blown out).
* \(\tau_{\text{past}, i}, \tau_{\text{future}, i}\): temporal horizons from autocorrelation of \(x_i(t)\).
* \(R_{\text{CLC}, i}\): spatial radius – how far along the ring strong correlations extend.

The **ring CLC** is:

\[
S_{\text{CLC, ring}} = \frac{1}{N} \sum_i S_{\text{CLC}, i},
\]

and as we ramp a global drive parameter \(\alpha\) (e.g., forcing amplitude \(F\) or effective gain \(\eta \sigma\)), we observe a **network cognitive knee**:

* small \(\alpha\): low CLC (system barely does anything),
* moderate \(\alpha\): high plateau of \(S_{\text{CLC, ring}}\) (rich, structured dynamics),
* large \(\alpha\): collapse of CLC as over-driven chaos destroys coherent patterns.

This gives us an operational definition of “how much thinking the ring is doing”.

### 2.4 Self-tuning ADRs: Lyapunov / coherence gates

To keep ADRs sitting on the high-CLC plateau, we add **slow plasticity**: a Self-Tuning Gate (STG) that gently adjusts parameters based on local chaos and coherence.

In the simulation chapters we used a practical controller:

* compute per-site **local coherence** \(R_{\text{local}, i}\) from phase synchrony with neighbours,
* choose a target band \([R_{\min}, R_{\max}]\) corresponding to “structured but not frozen”,
* let coupling strengths \(\kappa_i\) and resource gains \(\eta_i\) drift slowly:

\[
\dot \kappa_i = \varepsilon_\kappa (R^* - R_{\text{local}, i}), \qquad \dot \eta_i = \varepsilon_\eta (R^* - R_{\text{local}, i}),
\]

clamped to reasonable bounds.

Intuition:

* if a site is too noisy (\(R_{\text{local}, i}\) small), strengthen coupling and resource drive;
* if too laminar (\(R_{\text{local}, i}\) large), weaken them.

In more advanced versions, the gate uses **finite-time Lyapunov exponents** \(\lambda_i\) instead of coherence and enforces a **Lyapunov band** \([\lambda_{\min}, \lambda_{\max}]\). But even the coherence gate already shows:

> A self-tuned ADR ring can hold itself near the edge of chaos over a wide range of external drive, widening and flattening the cognitive knee plateau.

### 2.5 The Instrumented ADR (iADR) as a SACP module

Chapter X04 wraps this all up into a **concrete Python class** `InstrumentedADR`, which is the canonical ADR tile for the SACP suite.

Key features:

* State per site:
  * \(x_i\) (position),
  * \(v_i = \dot x_i\) (velocity),
  * \(r_i\) (resource),
  * \(k_i\) (plastic diffusive coupling).
* Dynamics: Euler-integrated version of the ADR equations plus optional global drive \(F \cos(\omega t)\).
* Plasticity: per-site coherence gate that updates \(k_i\) every few steps, pushing coherence toward a target \(R_{\text{target}}\).
* **SACP interface**:
  * *Sense*: exposes histories of \(x_i, v_i, r_i, k_i\), plus global and local coherence.
  * *Act*: accepts external control inputs \(u_i(t)\) added to \(\dot v_i\) (coupling to other tiles or controller agents).
  * *Couple*: internal topology and resource loop.
  * *Perturb*: supports shocks (state kicks) and parameter changes for lesion and robustness experiments.

Alongside this, we provide helper functions that:

* compute CLC proxies for logs,
* run **network knee experiments** (sweep drive \(F\), measure \(S_{\text{CLC, ring}}\)),
* compare **isolated vs coupled** regimes (showing autocatalytic gain),
* simulate **lesion and recovery**, demonstrating autopoietic behaviour.

This `InstrumentedADR` is exactly the thing you “drop autocatalytic chemistries into” in the SACP/ABCP suite: you plug in a mapping from biochemical variables to ADR sites, choose the drive parameter, and the platform takes care of integration, scanning, and instrumentation.

---

## Chapter 3 — From ADR Tiles to a Complexity Accelerator

The final piece is to put ADRs (and their cousins) back into the broader MAES and complexity-accelerator picture.

### 3.1 The Complexity Accelerator concept

In *The Architecture of Emergence* and *Coupled Learners at the Edge* we framed a general control panel for chaotic systems: a **Complexity Accelerator**.

Core ideas:

* treat each system as a **tunable attractor bundle** \((X, \Theta, F_\theta)\),
* instrument:
  * Lyapunov exponents,
  * transfer entropy,
  * avalanche and dwell statistics,
  * robustness to shocks;
* give an **agent/controller** (possibly an LLM) a cockpit to:
  * sweep parameters,
  * watch eigenvalues and information flow,
  * steer systems toward quasi-critical regimes (edges of chaos).

We implemented this first with:

* coupled logistic maps,
* coupled Lorenz oscillators with EFE-like controllers that self-select interior coupling and nonlinearity to maximise information flow at tolerable instability.

ADRs and ADR–Krebs tiles are **plug-compatible** with this architecture: they are just another \((X, \Theta, F_\theta)\) with a nicely interpretable \(\Theta\).

### 3.2 General pipeline: arbitrary autocatalytic chemistry → ADR tile

Given a new autocatalytic chemical network (metabolism, haemostasis, gene regulatory network), the pipeline is:

1. **Identify autocatalytic structure**:
   * positive feedback loops,
   * cycles that regenerate their own catalysts,
   * control-like variables (flux, redox, voltage, free-energy dissipation).
2. **MAES-ST embedding**:
   * define \(A, E, B, I\) for the pathway,
   * identify a small set of **order parameters** to keep (e.g., throughput, redox, potential, energy currency),
   * identify a key **context parameter** (e.g., energy demand, leak rate) that acts like Grytsay’s \(\alpha_4\).
3. **ADR surrogate construction**:
   * map order parameters to sites \(x_i\),
   * choose double-well potentials \(V_i\) where bistability is meaningful (healthy vs pathological, low vs high flux),
   * wire them into a ring with resource loop \(r_i\) following biochemical arrows,
   * tune parameters so that sweeping the chosen control parameter reproduces:
     * the period-doubling route to chaos,
     * approximate shape of attractor projections in key planes (as we did for the Krebs case).
4. **Instrument and integrate**:
   * wrap the resulting ADR in the `InstrumentedADR` class,
   * run cognitive-knee scans, lesion and recovery tests, and **autocatalytic gain** experiments (network vs isolated),
   * couple ADRs to other modules (e.g., self-tuning Potts blankets, small-world agents) via SACP.

Once this is done once for a given chemistry, that **ADR tile becomes the canonical representation** used everywhere else in the book and in your tools.

### 3.3 Boundaries, blankets, and higher-order agents

The ADR tile is the “internal world” piece. Chapters 20–23 build equally canonical **boundary tiles**:

* **Self-tuning Ising and Potts blankets** on small-world graphs,
* whose \(\beta\) drifts to a percolation-like regime where cluster-oracle mutual information peaks (maximal information flow across the blanket at tolerable order and disorder).

These blanket tiles:

* provide **rich, multi-colour semantics** (Potts clusters as proto-objects),
* are controlled by their own boundary STGs,
* serve as active Markov blankets in MAES agents.

Putting it all together:

* an **SW agent** has:
  * an internal chaotic controller (which can be an ADR ring),
  * a self-tuning Potts boundary,
  * and sits on a shared Potts field with other agents.

Clusters in that field become **shared meanings** that span agents; the blankets stay near criticality by self-tuning \(\beta\).

### 3.4 What we’ve actually achieved

Stepping back, the “chat so far” and the attached drafts have jointly built:

1. A rigorous **MAES-ST formalism** for context-dependent, stochastic, chaotic systems.
2. Detailed **case studies** showing:
   * metabolic networks (Krebs) as minimal cognitive engines with strange attractors;
   * gene regulatory networks as cooperative autocatalytic systems that can exhibit chaos when coupled to cellular state.
3. A **canonical ADR tile**:
   * defined mathematically,
   * interpreted physically,
   * implemented in code with instrumentation.
4. A **self-tuning mechanism** (STGs) to hold ADRs and blankets at the edge of chaos, defined both conceptually and in executable scaffolds.
5. A clear **integration story** into a SACP/ABCP-style Complexity Accelerator and multi-agent ecologies.

These three chapters are meant to be the “spine” for the Autocatalytic Duffing Ring primer: enough context for a reader (or another LLM) to understand what ADRs are, how they arise from classical chemistry, how they are implemented, and how they plug into the larger MAES/SACP machinery.

If you’d like, next step we can:

* zoom in on **one** ADR–Krebs parameter set and write a fully explicit worked example (figures, parameter tables, code snippets), or
* do the same distillation for **haemostasis** or **GRN motifs**, now that the template is clear.

---

## Appendix — Grytsay Hemostasis as an ADR Ring (detailed surrogate notes)

This appendix captures the extra detail from the ADR–Grytsay chat about recasting the 12-ODE hemostasis/atherosclerosis model as a concrete ADR surrogate and how to instrument it.

### A.1 Horizontal morphisms: the Grytsay flow as an ADR ring

* State \(X = (At, Tx, Ap, P, E_1, E_2, R, T_x^*, F, L, L^*, C)\) already behaves like a **ring of Duffing-like work units** with an autocatalytic loop \(L \to L^* \to C \to T_x^* \to L\).
* Relabel work variables \(x_i\) ↔ \(\{Tx, P, R, L, L^*, C\}\); resources \(r_i\) ↔ stocks \(\{F, L, L^*, C\}\); couplings \(\kappa_{ij}\) ↔ enzymatic/signalling arrows (Fig. 1 of the paper).
* Horizontal morphisms \(h_t\): the 12D ODE flow at fixed \(\mu_0\) falls onto fixed points, limit cycles, period-doubled cycles, or the strange attractor (Feigenbaum cascade around \(\mu_0 \approx 0.437\)).
* \(\mu_0\) plays the role of a **global drive/leak knob** exactly like an ADR’s effective gain; scanning \(\mu_0\) gives the same fixed→periodic→chaotic progression seen in ADR knee experiments.

### A.2 Metabolic Cognitive Light Cone (CLC) in hemostasis

* Grytsay’s Lyapunov spectra, KS entropy, and “foresight horizons” furnish the temporal part of a CLC.
* Define \(S_{\text{CLC, hemo}}(\mu_0)\) from:
  * identity lifetime \(C_\infty\): duration a hemostatic regime remains recognisable;
  * temporal horizons \(\tau_{\text{past}}, \tau_{\text{future}}\): Lyapunov times / foresight;
  * spatial radius \(R_{\text{CLC}}\): correlation span across \((P,R,Tx,T_x^*,L,L^*,C)\).
* Expect a **cognitive knee**: high CLC in the regular autooscillatory band, then a sharp drop when entering the mixing-funnel strange attractor as \(\mu_0\) is lowered.

### A.3 ADR surrogate design for hemostasis (4 sites)

Coarse-grain to four ADR sites with resource loop \(x_1 \to x_2 \to x_3 \to x_4 \to x_1\):

1. \(x_1\) Prostanoid balance (P vs Tx; R modulation). \(r_1\): prostanoid drive.
2. \(x_2\) Thrombus shock (T\(^*_x\)/R). \(r_2\): damage/clotting drive.
3. \(x_3\) LDL / plaque load (F, L, L\(^*\)). \(r_3\): plaque-derived inflammatory load.
4. \(x_4\) Cytokine field (C). \(r_4\): cytokine pressure back to prostanoids.

Equations (mod 4):
\[
\dot x_i = v_i,\quad
\dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \kappa (x_{i+1}+x_{i-1}-2x_i) + \eta_{i-1,i} r_{i-1} + F_i(t),
\]
\[
\dot r_i = -\mu_i r_i + \sigma_i x_i^2,
\]
with optional tilts \(b_i\) in \(V_i\) to bias healthy wells.

### A.4 Map \(\mu_0\) → plaque leak and scan

* Identify \(\mu_{\text{plaque}} \equiv \mu_3 \leftrightarrow \mu_0\) (plaque clearance); hold other \(\mu_i\) fixed.
* Scan \(\mu_{\text{plaque}}\) (e.g. 0.45→0.43) and build a bifurcation diagram from sampled \(x_3\) (plaque surrogate); target fixed→2→4→chaotic band like Grytsay’s \(L(t)\)/\(F(t)\) vs \(\mu_0\).

### A.5 Parameter skeleton (tunable)

* Potentials: \(a_1=a_3=1.0,\ a_2=a_4=0.8\); small biases \(b_i<0\) toward healthy wells.
* Damping/time scales: \(\gamma_{1,2}\sim0.3\) (fast prostanoid/thrombus), \(\gamma_{3,4}\sim0.1\) (slow lipid/cytokine).
* Couplings: \(\kappa\approx0.05\); \(\eta_{12}\sim0.6,\ \eta_{23}\sim0.7,\ \eta_{34}\sim0.8,\ \eta_{41}\sim0.4\).
* Resources: \(\sigma_i\approx0.5\); \(\mu_{1,2,4}=0.1\); \(\mu_{3}=\mu_{\text{plaque}}\) (scanned).

### A.6 Matching Grytsay projections

* Define surrogates: \(P_{\text{eff}}\sim x_1\); \(L^*_{\text{eff}}\sim \max(0,x_3)\); \(C_{\text{eff}}\sim \max(0,x_4)\); \(T^*_{\text{eff}}\sim \max(0,x_2)\); \(R_{\text{eff}}\sim -x_2\).
* At chaotic \(\mu_{\text{plaque}}\) (≈0.437), compare projections \((P,L^*), (P,C), (R,T^*)\) to Grytsay’s Fig. 4–5 for qualitative funnel shapes and banding; tweak \(\gamma_i, a_i, b_i, \eta_i, \kappa\) to match.

### A.7 Instrumentation and CLC experiment

* Use the instrumented ADR scaffold (N=4, per-site params) to:
  * sweep \(\mu_{\text{plaque}}\) and record sampled \(x_3\) (bifurcation),
  * compute CLC proxies (autocorrelation horizons, local coherence, spatial correlations) for each \(\mu_{\text{plaque}}\),
  * plot \(S_{\text{CLC, hemo}}(\mu_{\text{plaque}})\) and locate the hemostatic cognitive knee.
* Perturbation hooks: kicks to \(x_i\); jumps in \(\mu_{\text{plaque}}\) or \(\gamma_i\) (ischemia/reperfusion); lesion a link (\(\eta_{23}\)) to mimic ETC inhibition.

### A.8 Extensions and research directions

* **Self-tuning \(\mu_0\)**: add \(\dot\mu_0 = \varepsilon(\text{target CLC} - S_{\text{CLC}})\); healthy gate vs pathological gate breakdown.
* **Network of vessel tiles**: couple multiple hemostasis tiles via cytokine/inflammatory resources; look for network-level cognitive knees vs LDL load and lesion dynamics.
* **Automated coarse-graining**: learn a map from the 12D Grytsay trajectories to 4D ADR states via manifold learning/autoencoders, then fit ADR parameters to match the pushed-forward vector field.

---

## Appendix B — Canonical ADR chemical tile (general recipe + hemostasis L/C/P/T)

This appendix captures a generic ADR tile recipe you can drop onto any enzymatic network, plus a second hemostasis coarse-graining (L/C/P/T) aligned with the canonical ADR equations.

### B.1 Canonical ADR equations (chemical form)

For a ring of \(N\) sites (typically 4–6):

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i(x_i)
= \kappa_i (x_{i+1} + x_{i-1} - 2x_i) + \eta_i r_{i-1} + I_i(p),
\]
\[
\dot r_i = -\mu_i r_i + \sigma_i f_i(x_i),
\]

* \(x_i\): coarse “mode” (healthy vs pathological).
* \(r_i\): resource/flux/catalyst produced at site \(i\), donated to \(i+1\).
* \(\kappa_i\): diffusive coupling (mutual influence).
* \(\eta_i r_{i-1}\): autocatalytic feed-forward from predecessor.
* \(I_i(p)\): slow external drive depending on a control parameter \(p\).
* Potentials: \(V_i(x_i) = \tfrac14 x_i^4 - \tfrac{a_i}{2} x_i^2\) (double-well).
* Activity → resource: \(f_i(x_i) = x_i^2\) by default.

### B.2 Hemostasis as L/C/P/T tile (second coarse-grain)

Define four sites (indices mod 4): \(L \equiv 1, C \equiv 2, P \equiv 3, T \equiv 4\).

1. **Site L (plaques / LDL burden)**
   * \(x_L>0\): plaque-heavy.
   * \(r_L\): “plaques shedding inflammatory mediators” (drives C).
   * Dynamics: \(\ddot x_L + \gamma_L \dot x_L + \partial V_L = \kappa_L(x_C + x_T - 2x_L) + \eta_L r_T + I_L(\mu_0)\).

2. **Site C (cytokine field)**
   * \(x_C>0\): high inflammatory tone.
   * \(r_C\): cytokine flux acting on prostanoids.
   * Dynamics: \(\ddot x_C + \gamma_C \dot x_C + \partial V_C = \kappa_C(x_P + x_L - 2x_C) + \eta_C r_L\).

3. **Site P (prostacyclin / thromboxane / R)**
   * \(x_P<0\): prostacyclin-dominant; \(x_P>0\): thromboxane-dominant.
   * \(r_P\): vascular tone / NO / PGI2 signal.
   * Dynamics: \(\ddot x_P + \gamma_P \dot x_P + \partial V_P = \kappa_P(x_T + x_C - 2x_P) + \eta_P r_C\).

4. **Site T (thrombus state)**
   * \(x_T>0\): clot forming.
   * \(r_T\): flow disruption + plaque stress, fed back to L.
   * Dynamics: \(\ddot x_T + \gamma_T \dot x_T + \partial V_T = \kappa_T(x_L + x_P - 2x_T) + \eta_T r_P\).

Ring logic: \(L \to C \to P \to T \to L\). Cutting \(\eta_L\) breaks autocatalysis (mirrors killing chaos in the full model).

**Control parameter (\(\mu_0\)) mapping**

* Map Grytsay’s plaque dissipation \(\mu_0\) to the **resource leak** at L:
  \(\dot r_L = -\mu_L(\mu_0) r_L + \sigma_L x_L^2\), with \(\mu_L(\mu_0) = \mu_{L,0} + c(\mu_0 - \mu_0^{\text{ref}})\).
* Effective loop gain:
  \[
  G_{\text{loop}}(\mu_0) \sim
  \frac{\eta_C \sigma_L}{\mu_L(\mu_0)} \cdot
  \frac{\eta_P \sigma_C}{\mu_C} \cdot
  \frac{\eta_T \sigma_P}{\mu_P} \cdot
  \frac{\eta_L \sigma_T}{\mu_T}.
  \]
  Lower \(\mu_0\) → lower \(\mu_L\) → higher loop gain → Feigenbaum cascade to chaos, matching Fig. 2/4 in Grytsay.

### B.3 Generic ADR tile recipe for any enzymatic network

1. **Coarse-grain to 4–6 modes** around the dominant feedback/cycle (flux, cofactor(s), gradient/field, currency/buffer, regulator/damage).
2. **Install Duffing sites**: choose \(V_i\) (double-well if two regimes); set \((a_i,\gamma_i)\) for stability and timescale.
3. **Define resource loops**: \(\dot r_i = -\mu_i r_i + \sigma_i f_i(x_i)\); wire \(r_i \to x_{i+1}\) (sign via \(\eta_i\); cross-talk via \(\kappa_i\)).
4. **Attach control knobs**: pick 1–2 global parameters (supply, leak, demand, inhibitor). Make \(\mu_j(p)\), \(\eta_k(p)\) or drives \(I_i(p)\) affine in \(p\). Tune so the ADR ring reproduces the same regime sequence (steady → oscillatory → period-doubling → chaos) and similar attractor projections.

### B.4 “API” for the canonical chemical ADR tile

* **State:** \((x_i, v_i, r_i)_{i=1..N}\), \(N=4..6\).
* **Dynamics:**
  \[
  \dot x_i = v_i,\quad
  \dot v_i = -\gamma_i v_i + a_i x_i - x_i^3 + \kappa_i(x_{i+1}+x_{i-1}-2x_i) + \eta_i r_{i-1} + I_i(p),\quad
  \dot r_i = -\mu_i(p) r_i + \sigma_i x_i^2.
  \]
* **Semantics:** site = macro-module; ring edges = dominant feedback/cycle; \(p\) = experimental knob.
* **Fitting rule:** tune parameters so the tile matches the route to chaos vs \(p\) and the attractor projections of the detailed model or data.

### B.5 Research directions unlocked

* **Chemical cognitive knees:** instrument CLC + Lyapunov bands; locate the knee (max CLC) vs control parameter (e.g., \(\mu_0\)).
* **Universality classes:** cluster fitted tiles by \((a_i,\gamma_i,\kappa_i,\eta_i,\sigma_i,\mu_i)\), Lyapunov spectra, knee location.
* **Tissues as ADR lattices:** tile space with ADR rings; couple via Act channels (cytokines, stress, oxygen); study lesion propagation, repair, collective knees.
* **Data-driven fitting:** learn \(x_i\) from time-series (PCA/autoencoders); fit ADR (neural ODE constrained to Duffing + resource form); use ADR metrics as health descriptors.

---

## Appendix C — Autocatalytic Chemistry Dashboard v1 (ABCP/SACP integration)

This appendix captures the “ready to code v1” plan: plug canonical ADR tiles and Grytsay-inspired metabolic surrogates into the existing ABCP/SACP stack as first-class attractor models.

### C.1 What ABCP already has

* **ABTC engine**: integrates ODEs, computes chaos metrics (LLE), supports a `BaseDynamics` plugin interface.
* **Attractorhedron**: builds Ulam/transfer operators, extracts \(|\lambda_2|\), mixing rates, metastable sets, gate experiments.
* **Dash UI (ABCP)**: chaos cockpit with parameter sliders, LLE gauge, live phase plots.

### C.2 What’s missing for “drop any autocatalytic chemistry”

* **Chemistry ingestion**: adapters for SBML/YAML or hard-coded ODEs as `BaseDynamics`.
* **ADR/MAES semantics**: mapping chemistry variables to canonical ADR tiles (flux, redox, potential, energy).
* **Chemistry-centric metrics**: CLC proxies and “network cognitive knees” vs control parameters (ATP demand, dissipation).

### C.3 Architecture layers (inside ABCP)

1. **Core engine** (existing): ABTC + Attractorhedron.
2. **AutocatalyticChemModel plugins**: wrap chemical ODEs (SBML or hand-coded) as `BaseDynamics`.
3. **ADR + MAES semantics**: canonical ADR–Krebs tile (`InstrumentedADR` specialisation) and adapters to fit arbitrary chemistries to ADR rings.
4. **Instrumentation**: reuse LLE/Attractorhedron; add CLC proxies per site/chemistry; tag MAES morphisms (horizontal/η_SA/vertical).
5. **UI “Chemistry” tab**: model selector (Krebs, ADR–Krebs, Hemostasis…), control knob slider, phase plots, LLE gauge, CLC bar, bifurcation strip, Attractorhedron button.

### C.4 Two v1 models to register

#### Model 1: `krebs_v1` (4D metabolic surrogate)

State \( (P, H, V, A) \) = (TCA throughput, NADH load, \(\Delta\Psi\), ATP fraction).

Control parameter \(u\): “ATP demand / proton leak / NADH oxidation” (Grytsay’s \(k_{15}\)-like knob).

ODE (dimensionless toy to be tuned):
\[
\begin{aligned}
\dot P &= a_1 P (1 - P^2) - b_1 H + c_1 u,\\
\dot H &= a_2 P - b_2\,k15_{\text{eff}}\, H (1 - V),\quad k15_{\text{eff}} = 1 + k15_{\text{slope}}(u-1),\\
\dot V &= a_3 H - b_3 V - c_2 V(1 - A),\\
\dot A &= a_4 V(1 - A) - b_4 u\,A,
\end{aligned}
\]
with defaults:
* \(a_1=1.0, b_1=0.6, c_1=0.3\)
* \(a_2=1.5, b_2=0.8, k15_{\text{slope}}=0.5\)
* \(a_3=1.2, b_3=0.7, c_2=0.4\)
* \(a_4=1.1, b_4=0.9\)
* baseline \(u=1.0\)

Register as `BaseDynamics` (`name="krebs_v1"`, `state_dim=4`, `rhs=krebs_v1_rhs`, plus defaults).

#### Model 2: `adr_krebs_v1` (4-site ADR surrogate)

State \(X = (x_1..x_4, v_1..v_4, r_1..r_4)\) (length 12), with
* \(x_1\): TCA throughput, \(x_2\): NADH, \(x_3\): \(\Delta\Psi\), \(x_4\): ATP.
* Resources: \(r_1 \to\) NADH, \(r_2 \to\) \(\Delta\Psi\), \(r_3 \to\) ATP, \(r_4 \to\) TCA (closing the ring).

Dynamics (indices mod 4):
\[
\dot x_i = v_i,
\]
\[
\dot v_i = -\gamma v_i + a_i x_i - x_i^3 + \kappa (x_{i+1}+x_{i-1}-2x_i) + \eta\, r_{i-1},
\]
\[
\dot r_i = -\mu_{\text{eff}} r_i + \sigma x_i^2,\quad \mu_{\text{eff}} = \mu_0(1+u),
\]
with defaults:
* \(a = (1.0, 1.1, 0.9, 1.05)\), \(\gamma=0.2\), \(\kappa=0.3\), \(\eta=0.6\),
* \(\mu_0=0.1\), \(\sigma=0.5\), control \(u\) as bifurcation knob.

Register as `BaseDynamics` (`name="adr_krebs_v1"`, `state_dim=12`).

### C.5 UI hooks and metrics

* Sliders: `u` (bifurcation), optional `eta` (autocatalytic strength), `kappa` (diffusive coupling), fuel `F`.
* Plots: phase projections ((NADH, ATP), (NADH, \(\Delta\Psi\))), LLE gauge, tiny bifurcation strip vs `u`, CLC bar from ADR/chemistry trajectories.
* Attractorhedron: button to build Ulam operator and show \(|\lambda_2|\), metastable sets, gate experiment.

### C.6 Coding sequence

1) Implement `adr_krebs_v1` using the existing `InstrumentedADR` scaffold; expose `u`.  
2) Implement `krebs_v1` ODE plugin; expose `u`.  
3) Add CLC endpoints in ABTC (reuse ADR CLC proxy).  
4) Add ABCP “Chemistry” tab with model selector + slider + plots.  
5) (Later) Wrap full Grytsay Krebs 19D ODE as `krebs_full` for calibration against ADR–Krebs.
