# Chapter 17 — Autocatalytic Duffing Rings

## 0. What this chapter is about

Earlier chapters treated strange attractors in “one place at a time”: single Lorenz flows, Hopfield reservoirs, MAES droplets, metabolic cycles.

This chapter introduces a new building block:

> **Autocatalytic Duffing Rings (ADRs)** — cyclic networks of non-identical Duffing-like elements that pump energy and information around a ring, self-tune to the edge of chaos, and act as minimal “engines of cooperation”.

We’ll:

- Start from a single Duffing-like element as a minimal nonlinear “work unit”.
- Close these elements into an autocatalytic ring where each site drives the next.
- Show how ring coupling produces collective strange attractors and “ring cognition”.
- Wrap the ring in a Lyapunov self-tuning gate (from the Self-Tuning Attractors chapter) to keep it near the edge of chaos.
- Sketch how a tissue or NK-topology of such rings recovers something very close to a bioelectric field with a target anatomical shape as an attractor.
- Reinterpret ADRs through the MAES-ST lens as minimal cognitive engines with semantic content.

The goal is not a fully realistic biological model, but a canonical toy system that captures the geometry of “many agents in a ring doing work for each other” using Duffing-style chaos.

---

## 1. From Duffing oscillators to catalytic work units

### 1.1 A reminder: the Duffing oscillator

A standard driven, damped Duffing oscillator is

\[
\ddot x + \gamma\,\dot x + \alpha x + \beta x^3 = F\cos(\omega t),
\]

with:

- \(\gamma > 0\): damping,
- \(\alpha, \beta\): linear and cubic stiffness (double-well for \(\alpha < 0, \beta > 0\)),
- \(F, \omega\): drive amplitude and frequency.

In potential form,

\[
\ddot x + \gamma\,\dot x + \partial_x V = F\cos(\omega t), \qquad
V(x) = \tfrac12 \alpha x^2 + \tfrac14 \beta x^4.
\]

For certain parameter ranges, trajectories live on a strange attractor: bounded, fractal, mixing, with positive Lyapunov exponent(s).

### 1.2 Generalized Duffing-like elements

We don’t want all agents to be identical “W-shaped” units. So we define a Duffing-like site \(i\) by:

- State \(x_i(t) \in \mathbb R\) (or low-dimensional \(\mathbb R^{d_i}\) if desired),
- A local potential \(V_i(x_i)\) with at least cubic nonlinearity,
- Damping \(\gamma_i\),
- Site-specific drive \(I_i(t)\).

Canonical form:

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i = I_i(t). \tag{1}
\]

Examples:

- Double-well Duffing: \(V_i(x) = \tfrac14 x^4 - \tfrac{a_i}{2} x^2\) (two metastable “opinions”).
- Single-well soft spring: \(V_i(x) = \tfrac12 k_i x^2 + \tfrac14 b_i x^4\).
- Asymmetric sites: add cubic term \(c_i x^3\) to bias one well.

Think of each site as a nonlinear work unit that can store energy (in its potential wells), switch between configurations, and phase-lock or chaos-lock to its inputs.

---

## 2. Building an Autocatalytic Duffing Ring (ADR)

### 2.1 Ring architecture

Take \(N\) such sites arranged on a ring. Indices are modulo \(N\): \(i+1 \equiv 1\) when \(i = N\), etc.

We introduce:

- State at site \(i\): \(x_i(t)\),
- Local resource / catalyst: \(r_i(t)\),
- Neighbor coupling via a discrete Laplacian,
- Autocatalytic coupling where resource from site \(i\) drives site \(i+1\).

The basic picture: site \(i\) produces \(r_i\) which drives site \(i+1\), with the ring closing at \(N \to 1\).

### 2.2 Dynamical equations for an ADR

A minimal ADR model:

\[
\ddot x_i + \gamma_i \dot x_i + \partial_{x_i} V_i = \kappa (x_{i+1} + x_{i-1} - 2 x_i) + \eta r_{i-1} + \xi_i(t), \tag{2}
\]
\[
\dot r_i = -\mu r_i + \sigma x_i^2. \tag{3}
\]

- \(\kappa\): diffusive coupling strength along the ring.
- \(\eta\): strength of catalytic drive from previous site’s resource \(r_{i-1}\).
- \(\mu > 0\): resource decay rate.
- \(\sigma > 0\): production rate (more activity → more resource).
- \(\xi_i(t)\): optional small noise.

Indices are taken modulo \(N\).

Interpretation:

- Each \(x_i\) is a Duffing-like oscillator pulled by neighbors and driven by the previous site’s resource.
- Each \(r_i\) integrates local activity and decays; it is the “chemical” that site \(i\) donates to \(i+1\).
- Without the ring (\(\eta = 0\)), we just get a coupled Duffing chain. With the ring, there is a genuine autocatalytic loop: energy injected at any point can circulate and amplify around the ring.

### 2.3 Autocatalysis in this language

Call the ADR autocatalytic if:

- The trivial rest state \(x_i = 0, r_i = 0\) is not globally attracting.
- A sufficiently localized kick at one site \(j\) produces a self-sustaining circulating pattern of \((x_i, r_i)\).
- Breaking the ring (remove the link between some \(i \to i+1\)) kills the sustained pattern.

Heuristically, we need:

- \(\sigma\eta\) big enough that the resource-driven forcing compensates damping,
- \(\kappa\) in a range where coupling spreads activity but does not homogenize everything to a fixed point,
- Nonlinear \(V_i\) parameter regimes that support oscillations / chaos when driven.

For uniform sites with \(V_i(x) = \tfrac14 x^4 - \tfrac{a}{2} x^2\), a rough condition for non-trivial activity is

\[
\eta \sigma \langle x^2 \rangle \gtrsim \gamma^2,
\]

so that the effective drive can overcome damping (\(\langle x^2 \rangle\) is a typical variance on the attractor).

---

## 3. Strange attractors in Autocatalytic Duffing Rings

### 3.1 From limit cycles to high-dimensional chaos

Even a single Duffing oscillator has familiar routes to chaos (period-doubling, crises, etc.). A ring of \(N\) heterogeneous sites behaves like a multi-Lorenz-style system: high-dimensional, many possible attractors, rich bifurcations.

As we tune a control parameter (e.g., global forcing, damping, or the product \(\eta \sigma\)), we typically see:

- Low drive: quiescent or small-amplitude limit cycles (ring behaves like a weakly coupled oscillator chain).
- Intermediate drive: traveling waves and phase-locked patterns around the ring.
- Higher drive: multi-site chaos — a high-dimensional strange attractor where energy and information circulate without settling into simple periodic patterns.

Let \(X = (x_1, \dot x_1, \dots, x_N, \dot x_N, r_1, \dots, r_N)\) be the full state. For parameter set \(\theta\), the flow \(\dot X = F_\theta(X)\) has an attractor \(A(\theta) \subset \mathbb R^{3N}\): sometimes a limit cycle (1D), sometimes a strange attractor (fractal, with positive Lyapunov exponents). The autocatalytic structure matters: the resource loop makes the ring self-exciting in a way analogous to metabolic cycles at chaos-capable regimes.

### 3.2 Lyapunov spectrum and “ring cognition”

Characterize a chaotic ADR by its Lyapunov spectrum \(\{\lambda_k\}_{k=1}^{3N}\), ordered \(\lambda_1 \ge \lambda_2 \ge \dots\):

- \(\lambda_1 > 0\): sensitive dependence on initial conditions (chaos).
- \(\sum_k \lambda_k < 0\): dissipation.
- Fractal dimension \(D_L \approx K + \tfrac{\sum_{k=1}^K \lambda_k}{|\lambda_{K+1}|}\) (Kaplan–Yorke).

Empirically (and by analogy with the metabolic case):

- Modest positive \(\lambda_1\) with several near-zero exponents produces edge-of-chaos dynamics: long transient memories, rich but structured variability, many metastable “micro-attractors” corresponding to quasi-stable patterns around the ring.
- Very large \(\lambda_1\) yields “thermal chaos”: too noisy to store structure.

We’ll turn this into a control target next: the ring will listen to its own Lyapunov spectrum and tune itself.

### 3.3 Heterogeneity is a feature, not a bug

Because each \(V_i\) can differ, the ring is a heterogeneous cooperative:

- Some sites act as flip-flops (deep double wells).
- Others as soft relays (single wells).
- Others as amplifiers (low damping, high nonlinearity).

The key insight: you don’t need identical Duffing units to get ring-level strange attractors. What matters is at least one path around the ring where the total gain of “drive → oscillation → resource → drive” exceeds 1, and enough nonlinearity and delay to break simple synchronization.

---

## 4. Self-Tuning ADRs and Lyapunov Gates

The previous chapter introduced Self-Tuning Gates (STGs) and Lyapunov bands: controllers that keep a system’s local chaos level inside a desirable range by adjusting parameters. The same idea slots naturally into ADRs.

### 4.1 Tunable ADR bundle and local chaos field

View the ADR as a tunable attractor bundle:

- State space \(X\) (all \(x_i, \dot x_i, r_i\)),
- Parameter manifold \(\Theta\) (damping, couplings, drive strengths, etc.),
- Flow family \(F_\theta : X \to X\).

Introduce a local chaos field

\[
\lambda : X \times \Theta \to \mathbb R^N,
\]

where \(\lambda_i(x, \theta)\) estimates a site-wise largest finite-time Lyapunov exponent around site \(i\). Concretely,

\[
\lambda_i(x,\theta) \approx \tfrac{1}{T} \log \frac{\|\delta x_i(T)\|}{\|\delta x_i(0)\|}
\]

over sliding windows of length \(T\), using tangent dynamics.

### 4.2 Lyapunov bands and the edge-of-chaos sweet spot

Pick a Lyapunov band \(B = [\lambda_{\min}, \lambda_{\max}]\) for “acceptable chaos”. For example:

- \(\lambda < \lambda_{\min}\): too laminar, not enough exploration.
- \(\lambda > \lambda_{\max}\): too chaotic, no memory.
- \(\lambda \in B\): “edge-of-chaos” where we want to live.

Define a penalty potential in Lyapunov space, for site \(i\):

\[
U_i(\lambda_i) = \begin{cases}
(\lambda_i - \lambda_{\min})^2, & \lambda_i < \lambda_{\min},\\
0, & \lambda_{\min} \le \lambda_i \le \lambda_{\max},\\
(\lambda_i - \lambda_{\max})^2, & \lambda_i > \lambda_{\max}.
\end{cases} \tag{4}
\]

Total Lyapunov cost: \(U(\lambda) = \sum_i U_i(\lambda_i)\).

### 4.3 Self-tuning dynamics for ADR parameters

Let \(\theta\) collect the tunable parameters of the ring (e.g., \(\gamma_1, \dots, \gamma_N, \kappa, \eta, \dots\)). A simple Self-Tuning Gate on \(\theta\) is a slow gradient flow:

\[
\dot \theta = -\varepsilon \nabla_\theta U(\lambda(x, \theta)), \quad 0 < \varepsilon \ll 1, \tag{5}
\]

much slower than the fast dynamics of \(x, r\).

In components, e.g., for per-site damping:

\[
\dot \gamma_i = -\varepsilon \frac{\partial U_i}{\partial \lambda_i} \frac{\partial \lambda_i}{\partial \gamma_i}. \tag{6}
\]

Heuristics:

- If site \(i\) is too laminar (\(\lambda_i < \lambda_{\min}\)): \(\partial U_i / \partial \lambda_i < 0\). If \(\partial \lambda_i / \partial \gamma_i < 0\) (more damping reduces chaos), then \(\dot \gamma_i > 0\): increase damping until clear of the laminar region, or adjust another parameter with opposite sign.
- If too chaotic (\(\lambda_i > \lambda_{\max}\)): \(\partial U_i / \partial \lambda_i > 0\), so \(\gamma_i\) drifts in the opposite direction, taming chaos.

Alternatively, control the resource coupling \(\eta\) and neighbor coupling \(\kappa\):

\[
\dot \eta = -\varepsilon \eta \; \nabla_\eta U \cdot \partial_\eta \lambda, \qquad
\dot \kappa = -\varepsilon \kappa \; \nabla_\kappa U \cdot \partial_\kappa \lambda.
\]

Finite-difference estimates suffice in simulation; no need for closed-form derivatives.

### 4.4 Lyapunov Gates as ring-level meta-dynamics

Conceptually, a Lyapunov Gate wrapped around an ADR:

- Observes (noisily) the chaos level at each site,
- Nudges parameters to keep the ring’s attractor in the chosen Lyapunov band,
- Implements a self-referential loop: the attractor modifies its own shape.

This gives the ring meta-stability: it can absorb parameter drift and environmental perturbations; it automatically pushes itself back toward regimes with “good computational properties” (memory + exploration); different rings (or different contexts) can prefer different bands.

In STAM language, the ADR plus Lyapunov Gate is a self-tuning attractor shard capable of maintaining itself near a desired universality class.

---

## 5. From rings to fields: tissues and NK-topologies

So far we considered a single ring. Biological tissues, gene regulatory networks, or NK landscapes look more like graphs of rings than isolated cycles.

### 5.1 ADRs on general graphs

Let \(G = (V, E)\) be a graph (e.g., lattice, fractal, or NK neighborhood graph). Put an ADR at each node \(v \in V\):

- Local states \(x_i(v), r_i(v)\),
- Local parameters \(\theta(v)\),
- Edges \(E\) define coupling between sites and between rings.

Generalize (2)–(3) to:

\[
\ddot x_i(v) + \gamma_i(v) \dot x_i(v) + \partial_{x_i} V_i(v) = \kappa \sum_{w \sim v} (x_i(w) - x_i(v)) + \eta \sum_j A_{ji} r_j(w) + \xi_i(v,t), \tag{7}
\]
\[
\dot r_i(v) = -\mu r_i(v) + \sigma f_i(x(v)). \tag{8}
\]

- \(w \sim v\) denotes neighbors in \(G\).
- \(A_{ji}\) is a wiring matrix for “which resource at \(w\) drives which site at \(v\)”.
- \(f_i\) can be \(x_i^2\), a threshold function, etc.

This produces rings within rings: node-internal rings (our ADR), and graph-level rings (feedback cycles across nodes).

### 5.2 Continuum limit: Duffing field equations

On a dense lattice, approximate the coupled chain by a field \(\phi(x,t)\) on a domain \(\Omega \subset \mathbb R^d\), with dynamics:

\[
\partial_t^2 \phi + \gamma(x)\,\partial_t \phi = D \nabla^2 \phi - \alpha(x) \phi - \beta(x) \phi^3 + J(x,t) + \eta \rho(x,t) + \xi(x,t), \tag{9}
\]
\[
\partial_t \rho = -\mu \rho + \sigma g(\phi). \tag{10}
\]

- \(\phi(x,t)\): coarse-grained “Duffing field” (e.g., voltage, morphogen, gene-expression order parameter).
- \(\rho(x,t)\): coarse-grained resource / catalyst density.
- \(D \nabla^2 \phi\): spatial coupling.
- \(\alpha(x), \beta(x)\): spatially varying local potential.
- \(J(x,t)\): external drive (injury, boundary conditions, patterning cues).
- \(g(\phi)\): production rule (e.g., \(g(\phi) = \phi^2\), a threshold, etc.).

This is a Duffing wave equation with reaction–diffusion resource coupling: the field can support local Duffing-like chaos, traveling waves, and pattern formation, with autocatalytic reinforcement via \(\rho\).

### 5.3 Target shapes as attractor states

Encode a target anatomical shape as a stable attractor of (9)–(10): choose \(\phi_\star(x)\) (e.g., target bioelectric pattern) and \(\rho_\star(x)\) consistent with ongoing maintenance. Design \(\alpha(x), \beta(x), J(x,t)\) and resource coupling so that

\[
\phi(x,t) \to \phi_\star(x), \qquad \rho(x,t) \to \rho_\star(x)
\]

for a wide set of initial conditions that share the same topology (same “body plan” but with wounds or perturbations). The Lyapunov-gate machinery can be used in parameter space to tune these fields toward regimes where the desired target becomes a robust attractor.

Conceptually:

- A wound is a local kick: \(\phi(x, t_0)\) and \(\rho(x, t_0)\) are disturbed.
- The coupled ADR field relaxes by moving downhill in a global energy landscape shaped by the Duffing potentials, spatial couplings, and resource dynamics.
- The attractor it falls back into is the morphological target.

This mirrors the MAES-ST view of morphological homeostasis as attractor dynamics in a context-dependent system.

### 5.4 Wound repair as attractor completion (sketch)

If a tissue implements something like (9)–(10):

- The intact body corresponds to a global attractor \(A_{\text{body}}\).
- A wound knocks the system to a different region of phase space but inside the same basin of attraction.
- The resulting relaxation path is a spatio-temporal strange attractor (especially if the Duffing nonlinearity is strong), but the endpoint is the original shape.

Autocatalytic Duffing Rings tiled over a tissue give a concrete, if stylized, model of how local nonlinear elements, coupled in rings and fields, driven by resource loops and Lyapunov-gated self-tuning, can implement something very much like a bioelectric field that “remembers” the correct shape and fills in the missing pieces.

---

## 6. ADRs through the MAES-ST lens

The MAES-ST framework describes Mirrored Agent-Environment Systems with stochastic dynamics, context sheaves, and semantic functors.

### 6.1 ADR as a minimal cognitive engine

Treat the ADR (or ADR field) as a MAES:

- Agent \(A\): internal ring states \(\{x_i, \dot x_i, r_i\}\).
- Environment \(E\): external drives, loads, and substrate (e.g., tissue or medium).
- Markov Blanket \(B\): boundary sites and couplings where the ring exchanges energy / information with the rest of the system.
- Interaction dynamics \(I\): the ADR equations plus Lyapunov gate, \(\dot s = F_U(s; \lambda_U) + \text{noise}\), where \(s = (A, E, B)\) and \(U\) is the current context (wound vs intact, different boundary conditions).

The ADR functions as a Minimal Cognitive Engine:

- It senses its environment via drives \(J, \eta \rho\) hitting the boundary sites.
- It acts back by injecting structured energy / resource patterns.
- It maintains its internal Lyapunov band, keeping itself in a good computational regime.

The self-tuning gate implements Dual-FEP-style adaptation: agent and environment co-minimize a free-energy-like functional that balances structure and variability.

### 6.2 Context, sheaves, and semantic emergence

In MAES-ST, each context \(U\) (e.g., different tissue regimes, injury states) has its own configuration \(M(U) = (A(U), E(U), B(U), I(U))\). ADR parameters and even topology can change across contexts:

- \(U_{\text{intact}}\): parameters tuned so that the body-plan attractor is strong and noise-robust.
- \(U_{\text{wound}}\): parameters and Lyapunov bands shift to favor exploratory repair dynamics — more chaotic, more plastic.
- \(U_{\text{scar}}\): later context where dynamics stabilize again.

The sheaf structure ensures these context-dependent ADRs glue together coherently as the system moves through contexts.

Now add a semantic functor \(S : M \to P_M\): it maps attractor structures of the ADR (limit cycles, chaotic shards, basin geometry) to meanings: “intact limb present”, “gap that needs filling”, “overgrowth to be suppressed”, etc. Lyapunov gates and autocatalytic loops correspond to semantic drives: they don’t just shape trajectories; they shape what the system can mean in each context.

The ADR thus becomes: a concrete dynamical system (Duffing ring with resource loops), a minimal cognitive agent (in MAES-ST), and a semantic engine whose strange attractors encode morphological and functional “beliefs” about its environment.

---

## 7. Summary and open directions

Autocatalytic Duffing Rings give us a compact playground where many strands of this book meet:

- Nonlinear dynamics: Duffing-like sites with double wells and cubic nonlinearity.
- Autocatalysis: resource variables \(r_i\) that circulate around a ring, turning local oscillations into a self-sustaining engine.
- Strange attractors: ring-level chaos, characterized by Lyapunov spectra and invariant measures, analogous to metabolic strange attractors.
- Self-tuning: Lyapunov Gates adjusting parameters to keep the ring at the edge of chaos, creating self-referential attractor bundles.
- Field limit: ADRs tiled over a graph or tissue yield Duffing–reaction–diffusion fields that can implement bioelectric-like patterning and wound repair by attractor completion.
- MAES-ST semantics: ADRs as minimal cognitive engines whose attractor geometry is their internal world.

Open directions (each could be its own chapter or code notebook):

- **Numerical experiments:** simulate small ADRs (e.g., \(N = 5\)) under varying \(\eta, \kappa, \sigma\); measure Lyapunov spectra and invariant measures. Implement a simple Lyapunov gate and show self-tuning from laminar to edge-of-chaos regimes.
- **ADR-based computation:** use ring states as reservoir features; train readouts to implement logic or pattern recognition. Compare performance at different Lyapunov bands.
- **Tissue-scale morphogenesis toy models:** discretize (9)–(10) on a grid; define a target shape; test wound-repair dynamics. Study how different spatial patterns of \(\alpha(x), \beta(x)\) encode different body plans.
- **Coupled ADR–MAES droplets:** wrap ADRs in MAES droplets (with explicit Markov blankets) and couple them to other realms (mechanical, chemical) via cross-realm coupling, as in MAES-ST.
- **Category-theoretic lifting:** treat ADRs as morphisms in a traced monoidal category (ring = trace), and Lyapunov gates as 2-morphisms that retune the morphisms themselves — bringing ADRs under the same categorical umbrella as metabolic cycles.

The main message: rings of cooperating nonlinear agents, even when individually messy and heterogeneous, can organize into self-tuning strange attractors that behave like tiny engines of cognition and repair. Autocatalytic Duffing Rings are a mathematically explicit, programmable version of that idea.
