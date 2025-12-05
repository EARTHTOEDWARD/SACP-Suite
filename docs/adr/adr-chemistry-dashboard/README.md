# ADR Chemistry Dashboard (context and design draft)

Let us treat Grytsay as if he accidentally built a biochemical Autocatalytic Duffing Ring (ADR) 30 years early. This document collects the contextual sections for the dashboard. Later sections will hold code; the rest are explanatory chapter-style notes we want to preserve.

---

## Primer insert

For the Autocatalytic Duffing Ring primer, drop in the condensed three-chapter recap here: [ADR–Grytsay Reinterpretation for the ADR Chemistry Dashboard](../chapters/x06-adr-grytsay-dashboard.md).

---

## 1. What Grytsay is actually modelling (compressed)

Grytsay's 2025 paper extends his earlier prostacyclin-thromboxane model to include LDL ("bad cholesterol"), plaque growth, and cytokine-driven inflammation in a blood vessel.

The model:

- Has **12 ODEs** for concentrations:
  - Arachidonic acid in endothelial cells and platelets (At, Ap),
  - Thromboxane Tx and prostacyclin P,
  - Enzymes E1, E2,
  - Guanylate cyclase R,
  - Inflammatory mediator (T_x^*),
  - Fat input F, LDL level L, plaque mass (L^*), cytokines C.
- Includes a strong **positive feedback loop**:
  - LDL (L) -> plaques (L^*) -> cytokines C -> more plaque growth (L^*) and inflammatory mediator (T_x^*) -> more thrombosis and further LDL uptake.
- Treats the **cholesterol dissipation rate** (mu_0) (removal of plaques) as a key control parameter.
- Shows a **Feigenbaum period-doubling cascade**: as (mu_0) is varied, the hemostasis process transitions from stable auto-oscillations to chaos via period doubling and an intermittent funnel strange attractor. Visible in phase-parametric diagrams, Fourier spectra, and attractor projections (Figs. 2-5).
- Computes Lyapunov spectra, KS entropy, and "foresight horizons", interpreting hemostasis as a dissipative system whose chaotic regimes correspond to pathological atherosclerosis.

So: a fully fledged nonlinear dynamical model with an explicit autocatalytic biochemical feedback loop. We will re-wrap that as an ADR tile.

---

## 2. Recasting hemostasis as an Autocatalytic Duffing Ring

### 2.1 The ring structure is already there

Redraw the kinetic scheme (Fig. 1) as a loop: a **ring of interacting work units**:

1. Arachidonic acid S uptake -> At, Ap.
2. At, Ap + enzymes E1, E2 -> Tx and P.
3. Tx and P antagonistically regulate guanylate cyclase R and platelet aggregation.
4. Destruction of plaques produces inflammatory mediator (T_x^*).
5. Fats F -> LDL L -> plaques (L^*) -> cytokines C -> more (T_x^*) and LDL uptake.
6. Cytokines + prostanoids/LDL feed back to the top.

Properties:

- fast local dynamics (enzyme kinetics, aggregation),
- slower stock variables (LDL, plaque, cytokines),
- directional resource circulation (fat, cholesterol, inflammatory energy).

This is the footprint of an ADR ring: sites (x_i) producing resources (r_i) that drive the next site.

### 2.2 Mapping to ADR sites and resources

ADR scaffold:
[
ddot{x}_i + gamma_i dot{x}_i + partial_{x_i} V_i(x_i) = kappa (x_{i+1} + x_{i-1} - 2 x_i) + eta r_{i-1} + xi_i(t),
]
[
dot{r}_i = -mu r_i + sigma x_i^2.
]

Minimal 4-site coarse-graining:

1. **Prostanoid balance site**
   - (x_1): Tx vs P.
   - V_1: double well ("Tx-dominated clotting" vs "P-dominated anti-clotting").
   - (r_1): effective guanylate cyclase R and platelet aggregation state.
2. **Thrombus / inflammatory shock site**
   - (x_2): (T_x^*) level.
   - (r_2): stored thrombosis/damage energy.
3. **LDL / plaque site**
   - (x_3): LDL L + plaque mass (L^*).
   - (r_3): long-lived plaque burden driving more (T_x^*) when ruptured.
4. **Cytokine field site**
   - (x_4): cytokines C.
   - (r_4): inflammatory micro-environment altering enzymes/receptors.

Resource flow:
[
r_1 -> x_2 -> r_2 -> x_3 -> r_3 -> x_4 -> r_4 -> x_1 -> ...
]

- Plaque (L^*) and cytokines C autocatalyse growth, matching (dot{r}_i = -mu r_i + sigma x_i^2).
- (eta r_{i-1}) = resource from previous stage drives next metabolic step.

Even a 4-site coarse-graining preserves the autocatalytic loop shape and timescale separation.

### 2.3 (mu_0) as the ADR drive / leak parameter

In Grytsay: (mu_0) = dissipation of plaque (L^*).

ADR view:

- (mu_0) ~ leak of the resource loop.
- Sweeping (mu_0) is like sweeping global drive/leak (alpha) in ADR knee experiments.

The L,F vs (mu_0) diagram is an ADR-like knee scan: quiescent -> structured -> chaotic via period-doubling.

---

## 3. New insights from Grytsay through ADR + CLC

### 3.1 Hemostasis as a ring-level cognitive engine

- A vessel segment is an ADR tile:
  - maintains metastable hemostasis,
  - senses plaque burden and shocks via LDL, L^*, C, T_x^*,
  - responds with Tx/P oscillations to modulate thrombosis.
- Lyapunov "foresight horizons" are the temporal part of CLC.
- Add spatial horizon (coupled tiles) and identity lifetime (how long a healthy regime persists).

### 3.2 Cognitive knees, not just transition to chaos

- Band in (mu_0) with structured oscillations, finite Lyapunov exponents, maximal CLC.
- Below: damped, sluggish, low CLC.
- Above: chaotic funnel, collapsed CLC, pathological.

### 3.3 Plaques and cytokines as slow structural memory (vertical dynamics)

- Stocks (L^*, C) are slow structural variables storing damage/diet history and reshaping potentials.
- Interpret as frozen-in vertical parameters; promote to plastic couplings (like k_i plasticity).
- Healthy: oscillations maintain low plaque/C, keeping couplings in high-CLC band.
- Pathological: plaque growth rewires couplings, reducing CLC.

### 3.4 Self-tuning vs. failure of vertical gating

Biology adapts (mu_0); Grytsay treats it as external. ADR language: make (mu_0) dynamic with a Lyapunov/CLC gate:
[
dot{mu_0} = epsilon (target_CLC - S_{CLC, ring}).
]

Healthy: working gate keeps (mu_0) near knee. Atherosclerosis: gate stuck in pathological band.

---

## 4. Concrete directions

### 4.1 Build an ADR surrogate of hemostasis

- 4-6 sites as above.
- Double-well potentials per site.
- ADR resources with feed-forward (eta r_{i-1}).
- Tune to match period-doubling onset and attractor projections vs (mu_0).

### 4.2 Apply CLC and knees to Grytsay trajectories

- For each variable, compute autocorrelation horizons, local coherence, CLC scores.
- Plot CLC vs (mu_0); compare to Lyapunov/KS entropy.
- Expect CLC maximum before full chaos.

### 4.3 Add explicit self-tuning (mu_0) gate

- Extend 12-ODEs with:
[
dot{mu_0} = epsilon (f_target - f_obs),
]
  where f_obs is coherence, finite-time Lyapunov, or CLC.
- Study convergence, meta-attractors, robustness to shocks.

### 4.4 Network of vessel-tiles = ADR tissue

- Each segment = Grytsay module or ADR surrogate with its own (mu_0).
- Coupling via inflammatory resource; possible shared systemic (mu_0).
- Questions: network-level knee vs LDL intake; lesion effects; self-tuned collective health.

### 4.5 Physical / synthetic chemistry experiments

- Build 3-4 coupled oscillators (BZ or enzymatic) mirroring the loop.
- Resource via chemostats/light control.
- Microcontroller as Lyapunov gate adjusting a flow rate (mu_0 analogue).
- Look for cognitive knee; add feedback to stay near it.

### 4.6 Link to other metabolic cycles

- Apply template to Krebs, glycolysis, gluconeogenesis.
- Identify autocatalytic loop, coarse-grain to ADR ring, compare CLC/knee structure.
- Build a library of metabolic ADR tiles (Krebs-ADR, glycolysis-ADR, hemostasis-ADR...).

---

## 4.1 Horizontal morphisms: Grytsay hemostasis as an ADR ring

### 4.1.1 From kinetic scheme to work-ring

State:
[
(At, Tx, Ap, P, E_1, E_2, R, T_x^*, F, L, L^*, C).
]

Each ODE mixes driving, saturation, and decay (mu_0). Positive feedback loop:
[
L -> L^* -> C -> T_x^* -> L.
]

Thromboxane/prostacyclin/guanylate triangle (Tx,P,R) is a competing pair of work units. Relabel:

- work variables x_i <-> Tx, P, R, L, L^*, C.
- resources r_i <-> F, L, L^*, C.
- couplings kappa_ij <-> enzymatic/signalling interactions.

The vessel is an ADR ring of biochemical modules; LDL-plaque-cytokine is the resource loop.

### 4.1.2 Horizontal evolution (h): ODE flow at fixed structure

For fixed parameters theta (k_i, alpha_i, mu, mu_0):

- At given (mu_0) the system relaxes to equilibrium, limit cycle, or in (0.43,0.438) a period-doubled strange attractor.
- Phase-parametric diagrams L(t), F(t) vs (mu_0) are horizontal-flow fingerprints.
- (mu_0) acts like global drive/dissipation; sweeping yields fixed -> limit cycles/tori -> strange attractors.

### 4.1.3 Horizontal CLC for a vessel

Use Grytsay's Lyapunov spectrum, KS entropy, foresight horizons, fractal dimension to define a metabolic CLC:

- Identity lifetime ~ persistence of hemostatic pattern.
- Temporal horizon ~ Lyapunov time.
- Spatial radius ~ co-moving components (P,R,Tx,T_x^*,L,L^*,C) correlations.

Plot S_CLC,hemo(mu_0): high in autooscillations, sharp change near Feigenbaum window, contraction in chaos.

### 4.1.4 Concrete reinterpretation / experiment sketch

1. Treat key loops as ADR sites (Tx+E1, P+E2, R, (T_x^*,C), (F,L,L^*)).
2. Define ADR-style CLC proxies on reduced trajectories.
3. Sweep (mu_0) and build S_CLC,hemo(mu_0); compare to phase diagrams, spectra, foresight horizons.

Expectation: high CLC plateau in autooscillations; metabolic cognitive knee as (mu_0) pushes into funnel attractor.

### 4.1.5 Directions opened

1. ADR reduction of biochemical networks (fit Duffing potentials, match transitions and Lyapunov spectra).
2. Metabolic edge-of-chaos tuning (coherence-based plasticity as hypothetical homeostasis).
3. Comparative CLCs across Grytsay models (Krebs, glycolysis, metabolism).
4. Coupling to MAES-ST and vertical morphisms (adaptation of parameters/couplings to drift toward high-CLC bands).

---

## 5. 4-site ADR surrogate for hemostasis (coding sketch)

### 5.1 Coarse-graining into 4 sites

12 ODEs split into fast prostanoid/thrombosis (At, Tx, Ap, P, E1, E2, R, T*_x) and slow lipid/inflammation (F, L, L*, C). Control:
[
dot{L}^* = mu_1 L L^* /(1+L+L^*) - mu_0 L^*.
]

Compress to 4 sites:

1. x_1: Prostanoid balance (P vs Tx, via R).
2. x_2: Thrombus shock (T*_x + R).
3. x_3: LDL/plaque load (F, L, L*).
4. x_4: Cytokine field (C).

Ring: x_1 -> x_2 -> x_3 -> x_4 -> x_1.

### 5.2 ADR equations for the ring

Indices mod 4:
[
ddot{x}_i + gamma_i dot{x}_i + partial_{x_i} V_i(x_i) = kappa_i (x_{i+1} + x_{i-1} - 2 x_i) + eta_i r_{i-1} + I_i,
]
[
dot{r}_i = -mu_i r_i + sigma_i x_i^2.
]

Double wells:
[
V_i(x) = 0.25 x^4 - 0.5 a_i x^2 + b_i x.
]

Semantics and arrows:

- r_1 -> x_2 (prostanoid imbalance affects thrombus risk).
- r_2 -> x_3 (damage promotes plaques).
- r_3 -> x_4 (plaques promote cytokines).
- r_4 -> x_1 (cytokines bias prostanoids toward Tx, lower R).
- Diffusive terms smooth neighbours.

### 5.3 Mapping (mu_0) -> ADR plaque leak

Set plaque resource decay to mu_plaque = mu_3 = mu_0 analogue:
[
dot{r}_3 = -mu_plaque r_3 + sigma_3 x_3^2.
]

Scan mu_plaque in [mu_hi, mu_lo]; lower mu_plaque = slower clearance -> higher excitation.

### 5.4 Concrete parameter skeleton

- dt ~ 0.01
- a_i = 1.0; gamma: (0.3, 0.3, 0.1, 0.1); kappa = 0.3; eta = 0.5; sigma = 0.5; mu others = 0.1; mu_3 scanned.
- Ring gain ordering: eta_1 ~0.4, eta_2 ~0.6, eta_3 ~0.7, eta_4 ~0.8.
- Healthy bias: b_i < 0 to prefer healthy wells.

### 5.5 What to match

Period-doubling vs mu_plaque:

- For mu_plaque in [0.45,0.43], integrate, discard transient, sample x_3, plot bifurcation.
- Tune eta, kappa, a_i/b_i to see fixed -> 2 -> 4 -> chaos like Grytsay's L(t) vs mu_0.
- Define L_eff = L_0 + c_L max(0, x_3) if needed.

Attractor projections:

- P_eff = P_0 + c_P x_1.
- L*_eff = L*_0 + c_L* max(0, x_3).
- C_eff = C_0 + c_C max(0, x_4).
- T*_eff = T*_0 + c_T* max(0, x_2).
- R_eff = R_0 - c_R x_2 (or function of x_2, r_2).

Plot at chaotic mu_plaque ~ 0.437 to mimic Grytsay's projections.

### 5.6 Implementation path (use iADR scaffold)

1. Set N=4; allow per-site a, gamma, k; add per-site eta, mu, sigma.
2. mu[2] (0-based) = mu_plaque to scan.
3. Add small biases b_i in acceleration.
4. Reuse experiment_network_knee pattern: scan mu_plaque, measure sampled x[:,2] instead of S_ring.

Outputs: mu_plaque bifurcation, chaotic trajectories, time series to map to P,L*,C,R,T*_x planes.

### 5.7 Insights

- Hemostasis as an autocatalytic ring tile: four-node positive feedback P/Tx -> T*_x/R -> L/L* -> C -> back.
- (mu_0) as ring-level cognitive knee: lower leak raises drive until knee -> chaos.
- Fast-slow + double wells give funnel attractors (P-L, R-T*_x) naturally.
- Reusable tile across tissues; clone with different bands.
- Path to automated coarse-graining: learn 4D ADR state via manifold learning/autoencoders.

---

## Meta-attractor flows, fractal blankets, cosmology (research strands)

**Idea 1**: gates moving in coupling space (meta-attractor dynamics over parameters).  
**Idea 2**: parameters as fractal Markov blanket knobs.  
**Idea 3**: parameters as cosmological knobs controlling a droplet ecology.

### 1. Meta-attractor flows in coupling space (STAC + self-tuning gates)

#### 1.1 Core questions

- For tunable F_theta with parameter theta, what are attractors of gate dynamics theta_{t+1} = theta_t + pi(lambda_t, theta_t)?
- When does the region lambda_1(theta) in Lyapunov band B define a stable subsheaf of the attractor sheaf over Theta?
- How does probe-response capacity Delta I vary along gate-induced parameter trajectories?

Key pieces: attractor monodromy/sheaves (STAC), self-tuning gates as Lyapunov-aware controllers, probe-response oracles for logistic/Lorenz systems. Object: tunable attractor bundle (X, Theta, F) with attractor sheaf A over Theta and self-tuning gate G inducing flow on Theta.

#### 1.2 Short-term: toy pipelines

- Logistic & Lorenz testbeds: define families, attach STAC data on a grid (Lyapunov, glyph+CLC, probe-response MI), implement self-tuning gate pushing lambda_1 into band B and ascending Delta I, study gate trajectories in theta-space vs NK landscapes.
- Multi-Lorenz x Hopfield internal worlds: theta includes reservoir spectral radius/leak/couplings; measure regime count, wrinkliness, CLC, oracle capacity; study meta-attractors.

#### 1.3 Mid-term: Lyapunov bands as subsheaves

- Sheaf F over parameter space U excluding bifurcations.
- Band region U_B = {theta in U | lambda_1(theta) in [lambda_min, lambda_max]}.
- Conjecture: restriction F|_{U_B} locally constant; gate respecting band keeps meta-attractor in U_B.
- Use bifurcation cohomology H^1(U_B, F) and monodromy classes.

#### 1.4 Long-term: classify gate meta-attractors

- Taxonomy: band fixed-point gates, chaotic gate flows, multi-band itinerant gates.
- Positive geometry: embed Theta into Attractorhedron/FractalHedron bundle; gate flows trace paths.
- Design rules: when does edge-of-chaos also optimize Delta I? When do Lyapunov bands and NK fitness align/conflict?

### 2. Fractal blankets with self-tuned boundary oracles

Specialize Idea 1 to Markov blankets between systems tuned to criticality/fractality and high information.

#### 2.1 Core objects

- MAES tiles and connectdome (Agent <- Blanket -> Environment).
- Lorenz-Ising/Potts blankets, cluster/percolation structure, fractal blankets with alpha_IE <= D_B, self-tuning gates adjusting beta/fields/rewiring.

#### 2.2 Short-term: boundary diagnostics and Delta I maps

1. System: two Lorenz attractors coupled through AF Potts small-world layer.
2. Blanket parameters theta: beta, couplings/frustration, rewiring, field mix.
3. Measure geometry/capacity: D_B (fractal dimension of blanket state support), cluster stats, probe-response Delta I (probes = Lorenz drive stories; responses = cluster observables).
4. Boundary phase diagrams: surfaces where D_B peaks, percolation critical, Delta I maximal.

#### 2.3 Mid-term: design a fractal blanket gate

- Gate state = blanket parameters; signals = Lyapunov for boundary, cluster stats, Delta I gradients.
- Policy: keep beta near critical band (scale-free clusters, target D_B, high Delta I), balance fields, adjust rewiring for high-information interfaces.
- Run gate online; track theta(t), D_B(t), Delta I(t), CLC across blanket; characterize meta-attractor in parameter space (self-organized critical boundary? multiple regimes?).

#### 2.4 Theory: fractal blanket geometry

- Refined inequality: for gate keeping D_B and lambda_1 in bands, alpha_IE(theta) <= min(D_B(theta), D_1(mu_theta)); gate orbits raise alpha_IE until geometry-limited.
- Connectdome flows: gate induces flow on interface moduli; study fixed points/cycles/bifurcations.
- Cluster oracles as higher-order MAES tiles: monotonicity of Delta I under coarse-graining or gate ops; connect to Attractorhedron/FractalHedron.

### 3. Attractor cosmology with cognitive droplet ecology

Upscale to cosmology: parameters = cosmological controls (quench schedules, noise, drive); blankets = droplet boundaries; agents = droplets with CLC metrics.

#### 3.1 Core picture

- Landau-Ginzburg toy universe with quench r(t) -> droplet formation (domains of +/- phi_0).
- Droplets have cores, boundaries, halos; admit CLC via probe-response on edges.
- Agents/glyphs (S) with CLC decorations and internal worlds (multi-Lorenz x Hopfield).
- Boundaries as Markov blankets (MAES spans).
- Gates keep systems at droplet-friendly edges of chaos.

Program: treat droplets as S glyphs, wire circuits, study a self-tuning cosmological gate shaping droplet ecology.

#### 3.2 Short-term: from droplets to glyph ecology

1. Run LG universe across quenches/parameters.
2. Detect droplets; measure CLC metrics (tau_p, tau_f, R, C_infty); embed simple internal dynamics if desired.
3. Ecological observables: glyph distribution, interaction graph, lifetimes, merge/split events.

#### 3.3 Mid-term: cosmological self-tuning gate

- Theta: quench schedule, noise, drive (and maybe curvature).
- Attach chaos field (Lyapunov exponents), ecology score J (mean CLC or diversity).
- Gate G_cosmo adjusts theta to keep edge-of-droplet chaos and optimize J.
- Study meta-attractors in theta: parameter bands for long-lived, high-CLC ecologies; universe phases (no droplets, transient, stable ecology, chaotic foam); Big Bang as meta-attractor selection.

#### 3.4 Long-term: droplets with internal worlds, cluster blankets, connectdome

- Droplets with internal worlds (S x W glyphs).
- Potts-like fields on interfaces -> cluster blankets between droplets (MAES tiles between droplets).
- STAC over cosmological parameters: sheaves for PDE attractor, droplet attractors, internal worlds, boundary clusters; study monodromy under parameter loops.
- Ecological NK landscapes: discretize cosmological parameter space, fitness = ecology score J; compare NK landscape to self-tuning gate trajectories.

### 4. How they fit together

1. Meta-attractor flows (Idea 1): STAC + self-tuning gates over parameter spaces.
2. Fractal blankets (Idea 2): parameters = boundary physics; gates sculpt connectdome and info capacity.
3. Cosmology & droplet ecology (Idea 3): parameters = cosmological controls; gates sculpt attractor and glyph circuits.

Execution order: start Idea 1 (Duffing/Lorenz NK + STG), then Idea 2 (Potts blanket gate), then Idea 3 (cosmological gate).

---

## Plan for examples and chapters

- Anchor constructs in runnable experiments (Colab-style), reuse oracle/Lorenz/Ising/cosmology code.
- Use existing attractor-native pieces: STAC/monodromy/NK landscapes, MAES + Lorenz-Ising/Potts blankets + cluster oracles, LG cosmology + droplet CLC metrics, self-tuning gates.
- Use chapter drafts as skins over the labs: each idea = 1-2 worked examples + abstraction into glyphs/sheaves/invariants.

Worked example queue:

1. **Idea 1**: self-tuned Duffing NK walk (parameter grid, Lyapunov, CLC, Delta I; STG keeps lambda in band, ascends fitness).
2. **Idea 2**: self-tuned Lorenz-Potts boundary (cluster oracle + beta/field/rewiring gate; track D_B, Delta I).
3. **Idea 3**: self-tuned toy cosmos (LG quench + droplet CLC + gate on r(t)/noise/drive to maintain droplet ecology).

---

## Canonical ADR chemical tile (generic recipe)

### 1. Coarse-graining

Pick 4-6 macro-modes around the dominant autocatalytic loop: flux, cofactor/redox, gradient/field, currency (ATP/ADP), maybe regulator/inflammation.

### 2. Site dynamics

Duffing per site (double wells for distinct modes), plus diffusive coupling and autocatalytic resource drive (r_{i-1}).

### 3. Resources

dot{r}_i = -mu_i r_i + sigma_i f_i(x_i), typically f_i = x_i^2. Wire r_i forward along loop; inhibitory links via sign or extra terms.

### 4. Control parameters

Choose 1-2 global knobs (load/leak, drive). Express ADR parameters as functions of these; tune to match route to chaos and attractor projections of the full model.

### 5. Why useful

- Compute chemical CLC and knees.
- Compare universality classes across pathways/tissues via ADR parameters and Lyapunov/CLC signatures.
- Pathology as attractor failure (laminar, hyper-chaotic, broken autocatalysis).
- MAES-ST integration: ring as minimal cognitive engine; self-tuning gates as homeostatic policies.

### 6. ADR-Krebs tile snapshot (example)

State: x_1=TCA, x_2=NADH, x_3=DeltaPsi, x_4=ATP; resources r_i along same ring. Equations:
[
dot{x}_i = v_i,
]
[
dot{v}_i = -gamma_i v_i + a_i x_i - x_i^3 + kappa (x_{i+1}+x_{i-1}-2 x_i) + eta r_{i-1},
]
[
dot{r}_i = -mu_i(p) r_i + sigma_i x_i^2,
]
with control p mapping to ATP dissipation/drive; tune to match Grytsay's bifurcation geometry.

---

## Next coding steps (dashboard v1)

- Wrap ADR-Krebs as a new SACP/ABCP model (BaseDynamics plugin).
- Expose control slider (ATP dissipation/drive), plot phase planes (NADH vs ATP, TCA vs DeltaPsi), LLE gauge, small bifurcation strip.
- Add CLC proxies from instrumented ADR scaffold for metabolic knees.
- Optional: add Grytsay full model as a second plugin; cross-compare bifurcations and attractors with ADR surrogate.

---
