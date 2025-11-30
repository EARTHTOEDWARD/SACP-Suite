# FractalHedron₂ sketch

This note mirrors the “v1 script” from the spec so you can drop symbolic multifractal analysis directly into the Lorenz/Attractorhedron flow.

## Core helpers

`src/sacp_suite/modules/fractalhedron/core.py` contains the reusable primitives and the API now exposes `POST /fractalhedron/run` (same parameters: section hits, `coding_spec`, `coding_params`, `k`, `Q`).

| Helper | Purpose |
| --- | --- |
| `build_symbolic_sequence(section_hits, coding_spec="...")` | Convert section hits into symbolic itineraries. Supports `x_sign` (binary lobes), `quadrant_xz` (four-quadrant coding on the X/Z axes), and `radius_bins` (radial shells; pass `coding_params={"bins": [...], "labels": [...]}` to override thresholds). |
| `build_kgram_counts`, `normalize_kgram_probs` | Count/normalize k‑grams in the symbolic itinerary. |
| `fractal_moments`, `build_fractalhedron_k` | Construct FractalHedronₖ (probabilities, moments, D_q, symbolic scale ℓₖ). |
| `fractal_face_flags` | Identify faces: zero-probability words, approximate monofractal bands, near-max and near-zero dimensions. |
| `run_fractalhedron2_example` | Convenience for the Lorenz binary code with `k=2`, `q∈{0, 2}`. |
| `compute_fractal_penalty`, `agent_cost_with_fractal` | Tiny Agenthedron hook so you can add a symbolic multifractal smoothness penalty term inside J. |

## Minimal Lorenz example

```python
import numpy as np
from sacp_suite.modules.fractalhedron import run_fractalhedron2_example

# section_hits: (T, 3) array already available from Attractorhedron (Lorenz section).
section_hits = np.load("lorenz_section_hits.npy")
fh2, faces = run_fractalhedron2_example(section_hits)

print(fh2["D_q"])        # multifractal dimensions at q=0,2
print(faces["symbolic_zero_words"])
```

Swap `section_hits` for whatever you already store in Attractorhedron (e.g., the arrays sent to `build_ulam`). Once you have `fh2`, you can:

* log `D_q(ρ)` or `D_q(α)` alongside `|λ₂|`, `γ`, `entropy` to see where faces / factorisations line up,
* feed the penalty term into Agenthedron or Inferencehedron while scoring actions/policies,
* extend the coding spec to three or four symbols without touching the rest of the pipeline.

Example multi-symbol usage:

```python
sym_seq = build_symbolic_sequence(section_hits, coding_spec="quadrant_xz")
fh_quadrant = build_fractalhedron_k(sym_seq, k=2, Q=(0.0, 2.0))
```

## Agenthedron hook

```python
from sacp_suite.modules.fractalhedron import AgentCostTerms, agent_cost_with_fractal

terms = AgentCostTerms(FFE=0.4, FFIE=0.2, Yield=0.6, Risk=0.1)
J = agent_cost_with_fractal(terms, fh2, weight_fractal=2.0, target_D2=0.95)
```

That implements the “fractal penalty” described in the spec: `J = α·FFE + β·FFIE − γ·Yield + δ·Risk + ε·(D₂ − target)²`.
