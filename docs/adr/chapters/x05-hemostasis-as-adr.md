# Chapter X05 - Hemostasis as an Autocatalytic Duffing Ring

Treat Grytsay's hemostasis and atherosclerosis model as if it were a biochemical ADR tile built 30 years early. The goal is to compress the 12-ODE system into a reusable ADR ring, connect it to CLC and self-tuning language, and sketch concrete surrogate and experiment directions.

---

## 1. What Grytsay is modelling (compressed)

Grytsay's 2025 paper extends a prostacyclin-thromboxane model to include LDL, plaque growth, and cytokine-driven inflammation. The model:

- Has 12 ODEs: At, Ap, Tx, P, E1, E2, R, T_x*, F, L, L*, C.
- Contains a strong positive feedback loop: L -> L* -> C -> T_x* -> L.
- Uses cholesterol dissipation mu_0 (plaque removal) as a key control parameter.
- Shows a Feigenbaum period-doubling cascade versus mu_0, ending in an intermittent funnel attractor; Lyapunov spectra and KS entropy are computed.

It is already a nonlinear autocatalytic feedback loop; we re-wrap it as an ADR tile.

---

## 2. Recasting hemostasis as an ADR ring

### 2.1 The ring structure

Redraw the kinetic scheme as a loop of work units:

1. Arachidonic acid uptake -> At, Ap.
2. At, Ap + enzymes E1, E2 -> Tx and P.
3. Tx and P regulate R and aggregation.
4. Plaque destruction -> inflammatory mediator T_x*.
5. F -> L -> L* -> C -> more T_x* and LDL uptake.
6. Cytokines and prostanoids feed back upward.

Fast local dynamics, slow stocks (LDL, plaque, cytokines), directed resource circulation: exactly an ADR ring (sites x_i, resources r_i driving the next site).

### 2.2 ADR site/resource mapping (4-site coarse-grain)

ADR scaffold:
[
ddot{x}_i + gamma_i dot{x}_i + partial_{x_i} V_i(x_i) = kappa (x_{i+1}+x_{i-1}-2 x_i) + eta r_{i-1} + xi_i(t),
]
[
dot{r}_i = -mu r_i + sigma x_i^2.
]

Minimal sites:

1. Prostanoid balance (x_1: Tx vs P, double well; r_1: guanylate cyclase / platelet state).
2. Thrombus shock (x_2: T_x*; r_2: stored thrombosis/damage energy).
3. LDL/plaque (x_3: L + L*; r_3: plaque burden driving T_x*).
4. Cytokine field (x_4: C; r_4: inflammatory micro-environment).

Resource flow: r_1 -> x_2 -> r_2 -> x_3 -> r_3 -> x_4 -> r_4 -> x_1 -> ...

Plaque and cytokines autocatalyse their own growth (dot{r} = -mu r + sigma x^2); eta r_{i-1} is resource from the previous stage driving the next.

### 2.3 mu_0 as ADR leak

In Grytsay: mu_0 = plaque dissipation. ADR view: mu_0 is leak of the resource loop. Sweeping mu_0 = sweeping drive/leak alpha in ADR knee scans: quiescent -> structured -> chaotic via period-doubling.

---

## 3. Insights from ADR + CLC

### 3.1 Hemostasis as a cognitive engine

- One vessel segment = ADR tile maintaining metastable hemostasis.
- Senses plaque/inflammation (L, L*, C, T_x*), responds via Tx/P oscillations.
- Grytsay's Lyapunov foresight horizons are the temporal part of CLC; we add spatial horizon (coupled tiles) and identity lifetime.

### 3.2 Cognitive knees

- Band in mu_0 with structured oscillations, finite Lyapunov exponents, maximal CLC.
- Below: damped, low CLC. Above: chaotic funnel, CLC collapses (pathological).

### 3.3 Slow structural memory

- Stocks L*, C are slow structural variables shaping other potentials.
- Interpret as frozen vertical parameters; could become plastic couplings (like k_i plasticity).
- Healthy: oscillations keep L*, C low, couplings in high-CLC band. Pathological: plaque growth rewires couplings, reducing CLC.

### 3.4 Self-tuning vs gate failure

Real biology adapts mu_0; Grytsay treats it as fixed. ADR: make mu_0 dynamic with a Lyapunov/CLC gate:
[
dot{mu_0} = epsilon (target_CLC - S_{CLC, ring}).
]
Healthy: gate keeps mu_0 near knee. Atherosclerosis: gate stuck in pathological band.

---

## 4. Directions and experiments

### 4.1 ADR surrogate of hemostasis

- 4-6 sites as above; double wells per site.
- ADR resources with feed-forward eta r_{i-1}.
- Tune so period-doubling onset and projections (P,L*), (P,C), (R,T_x*) match Grytsay qualitatively.

### 4.2 CLC and cognitive knees on Grytsay trajectories

- For each variable/time series: autocorrelation horizons, local coherence, CLC scores.
- Plot CLC vs mu_0; compare to Lyapunov and KS entropy.
- Expect CLC maximum before full chaos; different variables peak at different mu_0.

### 4.3 Explicit self-tuning mu_0 gate

Add:
[
dot{mu_0} = epsilon (f_target - f_obs),
]
with f_obs = coherence, finite-time Lyapunov, or CLC. Study convergence, meta-attractors, robustness to shocks in F or C.

### 4.4 Network of vessel tiles

Each segment = Grytsay module or ADR surrogate with its own mu_0. Couple via inflammatory resource (C, T_x*); possibly share systemic mu_0. Questions: network knee vs LDL intake; lesion response; emergent collective health under self-tuning mu_0^{(i)}.

### 4.5 Physical / synthetic chemistry experiment

- Build 3-4 coupled oscillators (BZ or enzymatic) mapping the loop.
- Resource via chemostats or light control.
- Microcontroller as Lyapunov gate adjusting a flow (mu_0 analogue) based on coherence.
- Look for cognitive knee; add feedback to hold near it.

### 4.6 Other metabolic cycles

- Apply template to Krebs, glycolysis, gluconeogenesis.
- Identify autocatalytic loop, coarse-grain to ADR ring, compare CLC/knee structure.
- Build library of metabolic ADR tiles.

---

## 5. Next steps

- Build a 4-site ADR surrogate for a specific Grytsay trajectory (e.g., mu_0 = 0.437 strange attractor), propose parameters, and map each well biologically.
- Integrate this surrogate into the instrumented ADR scaffold for code experiments and knee scans.
