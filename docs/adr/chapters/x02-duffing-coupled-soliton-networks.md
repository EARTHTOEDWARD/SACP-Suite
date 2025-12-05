# Duffing–Coupled Soliton Networks as Autocatalytic Cognitive Systems

*Edward Farrelly; ChatGPT (LLM collaborator). Draft notes for the Autocatalytic Duffing Rings book.*

---

## Abstract

We sketch a minimal model in which mass–energy solitons (MES droplets) become the basic “cells” of a cognitive tissue when coupled through Duffing-style dynamic oscillators arranged in rings or small graphs. Each soliton is treated as a minimal cognitive engine with a core, edge, and halo, together with a quantifiable **Cognitive Light Cone** (CLC) measuring how far it can remember, predict, and act while remaining below a **cognitive knee** where coherence collapses.

When multiple solitons are connected by nonlinear Duffing couplers whose parameters can slowly yield, the resulting network can: (i) explore many joint attractor configurations, (ii) selectively stabilise those that improve the individual and collective CLCs, and (iii) maintain the very couplings that make this improvement possible. In this sense, the network is both **autocatalytic** (each member’s cognition is supported and enhanced by the others) and **autopoietic** (the organisation of couplings maintains itself by maintaining the solitons it depends on).

We describe a three-morphism perspective (fast horizontal dynamics, exploratory strange-attractor dynamics, and slow vertical yielding), give a minimal network-level formalisation, and outline simulation experiments to probe network-level cognitive knees and autopoietic behaviour.

---

## 1. Introduction

In the MES (Mass–Energy Solitons) framework, localised, self-maintaining field configurations are treated as minimal “Thinkers”: droplets of mass–energy that persist by balancing inflows and outflows at their boundary and by regulating exchanges with the environment. Each droplet has:

- a **core** storing bulk mass and internal energy,
- an **edge** or ring where sensing and acting occur,
- and a **halo** or wake that carries traces in the surrounding field.

From the outside, this looks like a tiny cognitive engine: it maintains its organisation over time and uses its boundary to regulate interaction with its environment.

We can summarise this “cognitiveness” by a **Cognitive Light Cone** (CLC), which measures how far into the past and future, and how far outward in space, the soliton can effectively couple while still holding itself together as a recognisable entity. There is a **cognitive knee**: beyond some level of drive or noise the droplet loses coherence and the CLC collapses.

**Question.** What happens when many such solitons are connected through structured nonlinear interactions? Can a network of solitons behave as a higher-order cognitive system, with autocatalytic and autopoietic properties?

We propose that **Duffing-style dynamic couplings**—rings of nonlinear oscillators whose own parameters can slowly adapt—offer a simple and expressive way to build such networks. The key idea is that the coupling tissue itself can exhibit oscillation, exploration, and yielding, and thus can implement a primitive learning loop over the space of interaction patterns between solitons.

---

## 2. MES solitons as minimal cognitive engines

Assume a field-theoretic setting where an MES soliton is a stable, localised solution of some underlying dynamics. We do not need the explicit equations here; we only assume:

- Each soliton \(i\) has a well-defined **core** region with internal state \(A_i\).
- Each soliton has an **edge** or blanket with observable state \(B_i(t)\), for example a ring-reduced phase profile or a compressed bit-stream carried by edge modes.
- The surrounding medium carries a **halo** \(H_i\) that encodes recent interaction history.

### 2.1 Cognitive Light Cone and cognitive knee

For each soliton define a **Cognitive Light Cone** score \(S_{\text{CLC},i}\) summarising its ability to remember, predict, and act:
\[
S_{\text{CLC},i} \approx C_{\infty,i}\,\sqrt{\tau_{\text{past},i}\,\tau_{\text{future},i}}\,R_{\text{CLC},i},
\]
where:

- \(C_{\infty,i}\) measures how long the soliton maintains a coherent identity,
- \(\tau_{\text{past},i}\) is the effective past horizon (how far back halo traces meaningfully influence the core),
- \(\tau_{\text{future},i}\) is the effective prediction horizon (how far ahead the soliton can stabilise trajectories),
- \(R_{\text{CLC},i}\) is an effective spatial radius of influence.

As drive intensity, noise, or external forcing increase, each soliton exhibits a **cognitive knee**: a relatively sharp transition beyond which \(C_{\infty,i}\) drops and the CLC collapses. Below this knee, the soliton functions as a minimal Thinker; above it, it is effectively reduced to noise and radiation.

---

## 3. Duffing oscillators and dynamic couplings

The Duffing oscillator is a classical nonlinear system with a double-well or multi-well potential and rich dynamical behaviour. In a simple form:
\[
\dot{x}_1 = x_2,\qquad
\dot{x}_2 = -\beta x_2 + \omega x_1 - \delta x_1^3 + f(t),
\]
with damping \(\beta\), linear stiffness \(\omega\), cubic nonlinearity \(\delta\), and external forcing \(f(t)\).

Depending on parameters, the system can exhibit:

- small oscillations in a single well,
- large-amplitude switching between wells,
- limit cycles and tori,
- chaotic motion with strong sensitivity to initial conditions.

In recent work, the couplings between multiple Duffing oscillators have themselves been made dynamic: coupling variables evolve according to their own differential equations, and their parameters change slowly in response to sustained patterns of activity. This reveals **sensitive windows** where small perturbations or changes in coupling structure can dramatically alter the global behaviour, allowing the system to function as a high-gain sensor or pattern recogniser.

We will treat such Duffing couplers as a convenient physical substrate for rich, nonlinear interactions between solitons.

---

## 4. Three morphism types: \(h\), \(\eta_{\text{SA}}\), \(v\)

To keep the conceptual structure simple, distinguish three types of evolution:

1. **Horizontal morphisms \(h\):** fast state evolution at fixed structure (ordinary relaxation and oscillation).
2. **Strange-attractor morphisms \(\eta_{\text{SA}}\):** regimes where chaos or quasi-periodic motion explores a larger region of state space.
3. **Vertical morphisms \(v\):** slow structural updates (yielding) of coupling parameters, encoding memory of what has worked.

Intuitively:

- \(h\): “fall into an attractor” (local optimisation),
- \(\eta_{\text{SA}}\): “shake around and visit alternatives” (exploration),
- \(v\): “nudge the structure to favour the good ones” (plasticity).

Learning is identified with repeated composition:
\[
X_{n+1} = v \circ \eta_{\text{SA}} \circ h (X_n),
\]
where \(X_n\) denotes the global state after \(n\) cycles.

---

## 5. Duffing–coupled soliton tissue

We now connect multiple MES solitons through Duffing-style dynamic couplings and interpret the resulting system as a simple cognitive tissue.

### 5.1 State space

Consider \(N\) solitons. For each soliton \(i = 1,\dots,N\):

- core state \(A_i\),
- edge state \(B_i\).

Connect solitons according to a graph \(\mathcal{E}\) (e.g., a ring where \((i,j)\in \mathcal{E}\) if \(j = i+1 \bmod N\)). For each edge \((i,j)\in \mathcal{E}\) introduce:

- Duffing state \(x_{ij} = (x_{ij,1}, x_{ij,2})\),
- structural parameters \(\eta_{ij}\) (effective coupling strength, phase, or nonlinearity).

The global state is
\[
X = \big( (A_i,B_i)_{i=1}^N,\; (x_{ij}, \eta_{ij})_{(i,j)\in\mathcal{E}} \big).
\]

### 5.2 Horizontal dynamics \(h\)

Horizontal dynamics describe fast evolution of soliton and coupler states at fixed structure:
\[
h_t : X \mapsto X(t),
\]
with
\[
\begin{aligned}
\dot{A}_i &= F_i(A_i, B_i; \text{local field}),\\
\dot{B}_i &= G_i(A_i, B_i, \{x_{ki}\}_{k:(k,i)\in\mathcal{E}}; \text{inputs}),\\
\dot{x}_{ij,1} &= x_{ij,2},\\
\dot{x}_{ij,2} &= -\beta x_{ij,2} + \omega x_{ij,1} - \delta x_{ij,1}^3 + g(B_i, B_j; \eta_{ij}),\\
\dot{\eta}_{ij} &= 0.
\end{aligned}
\]
Here \(F_i\) and \(G_i\) encode the MES soliton dynamics and their coupling to the Duffing variables \(x_{ij}\), while \(g\) describes how edge states drive the coupler.

### 5.3 Strange-attractor dynamics \(\eta_{\text{SA}}\)

For certain parameter regimes, parts of the network exhibit chaotic or multi-torus motion with a positive largest Lyapunov exponent. Denote that evolution by
\[
\eta_{\text{SA},t} : X \mapsto X(t) \quad\text{with}\quad \lambda_{\max}(X) > 0.
\]
These regimes allow the network to explore different coordination patterns between solitons: different phase relationships, switching patterns, and joint attractors.

### 5.4 Vertical dynamics \(v\)

Vertical dynamics describe slow structural updates of the Duffing couplers:
\[
v : X \mapsto X', \qquad \dot{\eta}_{ij} = \varepsilon\, G_{ij}(B_i, B_j, x_{ij}), \quad 0 < \varepsilon \ll 1.
\]
The function \(G_{ij}\) implements a plasticity rule. Examples:

- Hebbian-like: increase coupling when \(B_i\) and \(B_j\) show sustained correlation.
- Information-theoretic: adjust \(\eta_{ij}\) so that the mutual information between \(B_i\) and \(B_j\) stays near a desired target.

### 5.5 Learning loop

A single learning cycle consists of:

1. **horizontal evolution \(h\):** solitons and couplers relax within their current basins;
2. **strange-attractor evolution \(\eta_{\text{SA}}\):** selected parts of the network visit chaotic or quasi-periodic regimes and explore alternative patterns;
3. **vertical update \(v\):** coupler parameters yield slightly in response to patterns encountered.

Iterating this loop,
\[
X_{n+1} = v \circ \eta_{\text{SA}} \circ h (X_n),
\]
implements a simple form of Natural Induction at the network level.

---

## 6. Autocatalysis and autopoiesis

### 6.1 Network Cognitive Light Cone

Given individual CLC scores \(S_{\text{CLC},i}\) for each soliton, define a network-level CLC:
\[
S_{\text{CLC,network}} = \sum_{i=1}^N w_i\, S_{\text{CLC},i},
\]
with non-negative weights \(w_i\) (e.g., all ones or reflecting resource importance). The network is **coherently cognitive** when \(S_{\text{CLC,network}}\) exceeds a threshold; a **network cognitive knee** is the sharp transition of \(S_{\text{CLC,network}}\) as a global drive or noise parameter varies.

### 6.2 Autocatalytic cognitive set

The soliton–coupler system forms an **autocatalytic cognitive set** if:

1. **Embedded solitons outperform isolated ones:** for many solitons \(i\),
   \[
   S_{\text{CLC},i}^{(\text{network})} > S_{\text{CLC},i}^{(\text{isolated})}
   \]
   after learning.
2. **Improvement depends on network structure:** removing or weakening key couplings significantly reduces both \(S_{\text{CLC},i}\) and \(S_{\text{CLC,network}}\).
3. **Learning drives toward such configurations:** under repeated \(v \circ \eta_{\text{SA}} \circ h\), the network tends to structural patterns \(\{\eta_{ij}\}\) that increase \(S_{\text{CLC,network}}\).

Plainly, the solitons make each other “smarter”, and the coupling pattern that achieves this is stabilised by the overall dynamics.

### 6.3 Autopoietic cognitive tissue

The network behaves as an **autopoietic cognitive tissue** if, in addition:

- the organisation of couplings (who connects to whom, with which \(\eta_{ij}\)) is maintained by the activity it supports;
- disruptions (loss or weakening of a soliton or coupler) trigger reorganisation that restores a similar pattern of CLC and coupling structure.

Practically, this would show up if:

- perturbations pushing some solitons toward their cognitive knees induce network-wide adjustments that bring them back below those knees;
- the resulting new coupling pattern again supports elevated \(S_{\text{CLC,network}}\).

At that point, the Duffing-coupled ring of solitons qualifies as a simple autopoietic cognitive tissue: a system whose structure produces Thinkers whose activity regenerates and stabilises that very structure.

---

## 7. Experimental directions

### 7.1 Basic setup

- **Solitons:** simulate \(N = 3\)–\(5\) MES droplets in a 1D or 2D field, each tuned so that in isolation they sit comfortably below their individual cognitive knees.
- **Edge observables:** define a simple edge signal \(B_i(t)\) for each soliton (e.g., dominant edge mode amplitude, edge phase, or a compressed bitstream derived from the ring).
- **Duffing couplers:** for each pair \((i,j)\in\mathcal{E}\) in a ring, define a Duffing oscillator
  \[
  \dot{x}_{ij,1} = x_{ij,2}, \qquad
  \dot{x}_{ij,2} = -\beta x_{ij,2} + \omega x_{ij,1} - \delta x_{ij,1}^3 + g(B_i, B_j; \eta_{ij}),
  \]
  where \(g\) is a simple function of the edge signals.
- **Plasticity rule:** choose a slow update for \(\eta_{ij}\), e.g.
  \[
  \dot{\eta}_{ij} = \varepsilon \big( \text{MI}(B_i,B_j) - \text{target} \big),
  \]
  with mutual information computed over a sliding window.

### 7.2 Protocol and measurements

1. Initialise solitons and couplers with random phases and near-neutral \(\eta_{ij}\).
2. Apply occasional perturbations to selected solitons, mimicking environmental events.
3. Run many cycles of \(h\), intermittent \(\eta_{\text{SA}}\)-like excursions (e.g., via drive modulation), and \(v\).
4. Track:
   - individual CLCs \(S_{\text{CLC},i}(t)\),
   - network CLC \(S_{\text{CLC,network}}(t)\),
   - coupling parameters \(\eta_{ij}(t)\),
   - dependence of these on a global drive or noise parameter.

**Key signatures of success:**

- most \(S_{\text{CLC},i}\) increasing relative to isolated baselines after learning;
- sharp drops in \(S_{\text{CLC,network}}\) when key couplings are removed, followed by partial recovery via re-learning;
- a clear network-level cognitive knee as global parameters are varied.

---

## 8. Outlook

We propose a minimal way to view networks of MES solitons as autocatalytic, autopoietic cognitive systems by wiring them together with Duffing-style dynamic couplings that can explore and yield. The basic moves are:

- treat each soliton as a minimal Thinker with a quantifiable CLC and cognitive knee;
- treat couplings as dynamic nonlinear oscillators whose parameters implement memory;
- let learning emerge from the composition of fast dynamics (\(h\)), exploratory strange-attractor regimes (\(\eta_{\text{SA}}\)), and slow structural yielding (\(v\));
- evaluate success via improved individual and collective CLCs and the emergence of network-level cognitive knees.

Natural extensions include embedding such tissues into larger agent–environment architectures, exploring more complex coupling graphs, and studying how multiple tissues interact in a shared field, potentially giving rise to higher-order cognitive organisations.
