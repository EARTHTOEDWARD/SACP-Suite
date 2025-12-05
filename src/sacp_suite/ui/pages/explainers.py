from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import html

# Plain-English explainers surfaced on each tab.
# Keep copy short, focused on purpose, insights, data, and example use cases.
EXPLAINERS: Dict[str, Dict[str, Any]] = {
    "home": {
        "title": "Home — ask and route",
        "purpose": "Capture your goal, choose whether you already have data, and kick off ingest checks or a synthetic dataset plan.",
        "insights": [
            "Turns a natural-language ask into a structured task spec with confidence and mode (batch vs real-time).",
            "Shows quick ingest previews plus schema/row health checks.",
            "Proposes a synthetic dataset recipe when you need starter data.",
        ],
        "data": [
            "Plain-language prompts plus toggles for data status and latency tolerance.",
            "Optional preview rows or column hints to guide validation.",
        ],
        "cases": [
            "Spin up monitoring for control rooms or lab instruments.",
            "Prep clinical, EEG, or sensor streams for downstream modules.",
            "Scope dataset needs for financial or industrial backtests.",
        ],
    },
    "simulator": {
        "title": "Simulator — quick Lorenz sandboxes",
        "purpose": "Play with the Lorenz 63 system, inspect phase portraits/time series, and estimate Lyapunov exponents on runs or uploads.",
        "insights": [
            "Impact of ρ/σ/β/dt on stability and geometry.",
            "LLE for native runs or a one-column uploaded series.",
            "Side-by-side view of a shared dataset against the canonical Lorenz shape.",
        ],
        "data": [
            "Built-in ODE integrator with your parameters.",
            "Shared trajectory sent from Datasets or another tab.",
            "Single-column CSV uploads for LLE-only checks.",
        ],
        "cases": [
            "Check volatility regimes by running LLE on price or risk signals.",
            "Stress-test reactor or HVAC control loops in a chaotic sandbox.",
            "Validate robotics or drone dynamics against a known attractor.",
        ],
        "open": True,
    },
    "sheaf": {
        "title": "Sheaf — regime sweep",
        "purpose": "Sweep ρ across Lorenz, tag attractor classes, and spot where gluing fails (obstructions).",
        "insights": [
            "λ₁ vs ρ curve to see where chaos enters or leaves.",
            "CLC vs ρ to gauge signal richness per regime.",
            "Section intervals and obstruction notes for regime changes.",
        ],
        "data": [
            "ρ value list plus step/burn/dt settings for the built-in sweep.",
        ],
        "cases": [
            "Map safe vs chaotic windows for controllers or turbines.",
            "Locate stable bands before deploying trading factors.",
            "Explore bifurcations in bioelectric or climate surrogates.",
        ],
    },
    "attractorhedron": {
        "title": "Attractorhedron — data-driven operator",
        "purpose": "Build an operator from the latest trajectory and inspect |λ₂|, γ, and v₂ fields.",
        "insights": [
            "Operator stability summary (|λ₂|, γ).",
            "v₂ heatmap to flag stable/unstable basins.",
            "Readiness of the trajectory for downstream symbolic/fractal steps.",
        ],
        "data": [
            "Shared trajectory from Simulator or Datasets (needs at least X,Z columns).",
        ],
        "cases": [
            "Process phase mapping for industrial or chemical plants.",
            "Manifold snapshots for metabolic or bioelectric tiles.",
            "Anomaly screening on sensor or financial flows.",
        ],
    },
    "fractal_llm": {
        "title": "Fractal LLM — symbolic multifractals",
        "purpose": "Build a multifractal (FractalHedron₂) from a trajectory and run a small reservoir that can be driven by text.",
        "insights": [
            "D_q spectrum, zero-probability words, and constraint checks.",
            "Reservoir trajectories and switch traces tied to text prompts.",
            "Flags for near-max or near-zero dimension regions.",
        ],
        "data": [
            "Shared trajectory or cached operator output.",
            "Text prompt to steer the reservoir when no shared data is present.",
        ],
        "cases": [
            "Translate narratives into symbolic regimes for cyber-physical reports.",
            "Detect shifts in clinical speech or acoustic waveforms.",
            "Compare symbolic fingerprints across operating regimes.",
        ],
    },
    "cognition": {
        "title": "Cognition — memory and discriminability",
        "purpose": "Probe how long a chaotic system remembers inputs and how well it separates concepts.",
        "insights": [
            "Memory profile M(k) plus sample trial overlays.",
            "CLC summary (lag capture, radius, score).",
            "Discriminability metrics (accuracy, mutual information lower bound).",
        ],
        "data": [
            "Synthetic input patterns you specify (alphabet, patterns, trials).",
            "Default Lorenz-driven trials for quick checks.",
        ],
        "cases": [
            "Pick reservoir length for finance or demand forecasting.",
            "Check detectability of clinical or industrial motifs.",
            "Benchmark symbolic encoders for trading or telemetry streams.",
        ],
    },
    "datasets": {
        "title": "Datasets — curate and route",
        "purpose": "Browse bundled strange-attractor datasets, upload your own CSV, preview, and send trajectories into other tabs.",
        "insights": [
            "Scatter/preview of rows and quick hints on where to send them.",
            "Shared trajectory wiring so other tabs can reuse the same data.",
            "Upload status with returned dataset ids.",
        ],
        "data": [
            "Bundled Lorenz, Rossler, and embedded CSVs.",
            "User CSV uploads (single file, small preview shown).",
        ],
        "cases": [
            "Bootstrap experiments without connecting production data.",
            "Stage clinical, chemical, or financial time series for analysis.",
            "Ship synthetic attractors to downstream operator or reservoir tabs.",
        ],
    },
    "dcrc": {
        "title": "DCRC — distributed reservoirs",
        "purpose": "Run coupled chaotic reservoirs, visualize their trajectories, and inspect network structure.",
        "insights": [
            "3D trajectories per reservoir with coupling overlays.",
            "Network diagram of reservoir links sized by coupling.",
            "Prediction stats summary and sample traces.",
        ],
        "data": [
            "Internally generated reservoir trajectories.",
            "Optional overlay of a shared dataset for comparison.",
        ],
        "cases": [
            "Ensemble predictors for markets, demand, or load.",
            "Sensor-network stability and coupling studies.",
            "Compare multiple controller parameterizations quickly.",
        ],
    },
    "self_tuning": {
        "title": "Self Tuning — keep systems on target",
        "purpose": "Run live self-tuning sessions or batch demos that adjust spectral radius to hold a regime.",
        "insights": [
            "λ₁ and spectral radius trends over time.",
            "Regime classification as tuning progresses.",
            "Attractor snapshot for the current state.",
        ],
        "data": [
            "Self-generated Lorenz states with your step count.",
        ],
        "cases": [
            "Closed-loop control for reactors or energy systems.",
            "Keep hardware near edge-of-chaos for compute efficiency.",
            "Watch for drift in autopilot or regulator loops.",
        ],
    },
    "bcp": {
        "title": "BCP — bioelectric sectioning",
        "purpose": "Section a trajectory or sample curve to inspect structure in a cross-section.",
        "insights": [
            "Scatter of section points in the chosen plane.",
            "Intersection counts to gauge structural complexity.",
            "Notes on whether shared data or a sample curve was used.",
        ],
        "data": [
            "Shared trajectory (preferred) with at least two axes.",
            "Fallback synthetic curve when no shared data is available.",
        ],
        "cases": [
            "Bioelectric tissue or wound cross-sections.",
            "Identify gating planes in robotics or avionics traces.",
            "Spot anomalies in financial or sensor phase slices.",
        ],
    },
    "abtc": {
        "title": "ABTC — integrator playground",
        "purpose": "Integrate a Lorenz-like attractor via the ABTC RK4 helper or visualize a shared dataset.",
        "insights": [
            "3D trajectory and time-series views for the run.",
            "Sensitivity to dt and step count you choose.",
            "Shared-vs-generated note to track provenance.",
        ],
        "data": [
            "ABTC-generated Lorenz trajectory.",
            "Shared dataset reused for visualization.",
        ],
        "cases": [
            "Generate training traces for downstream models.",
            "Benchmark integrators against measured sensor tracks.",
            "Share trajectories to operator or reservoir tabs.",
        ],
    },
    "adr": {
        "title": "ADR — module directory",
        "purpose": "Browse ADR module variants (Krebs, hemostasis, bioelectric, bouquet) and their entry points.",
        "insights": [
            "Concise summaries of each ADR module.",
            "API endpoints and plugin keys for notebooks or services.",
        ],
        "data": [
            "No live data required; acts as a navigation card.",
        ],
        "cases": [
            "Metabolic tile simulation for pharma or biotech.",
            "Vascular/hemostasis scenario testing.",
            "Connect bouquet stack APIs into pipelines.",
        ],
    },
    "chemistry": {
        "title": "Chemistry — ADR bioelectric run",
        "purpose": "Run ADR bioelectric simulations with optional initial conditions and see trajectories per state.",
        "insights": [
            "Time-series per chemical/electric state with optional LLE.",
            "Quick readout of simulation metadata returned by the API.",
        ],
        "data": [
            "t_max, dt, and optional initial state vector.",
        ],
        "cases": [
            "Explore tissue potential dynamics and wound-healing surrogates.",
            "Stress-test parameter sets before fitting lab data.",
        ],
    },
    "frac_chem_sprott": {
        "title": "Frac Chem Sprott — hidden attractor",
        "purpose": "Simulate or scan a fractional-order chemical Sprott system with bifurcation, complexity, and current views.",
        "insights": [
            "Trajectory, phase portrait, and spectral entropy/complexity grids.",
            "Bifurcation plots vs q or k1, highlighting stability bands.",
            "Current decompositions (JS/JC/JR/JE) when returned.",
        ],
        "data": [
            "Fractional order q, k1, offset m, initial conditions, and scan ranges.",
        ],
        "cases": [
            "Study fractional kinetics stability windows.",
            "Hunt for hidden attractors in chemical or bioelectric surrogates.",
            "Plan parameter sweeps for control or identification.",
        ],
    },
    "bouquet": {
        "title": "Bouquet CLC — stack scans",
        "purpose": "Run or scan bouquet stacks to probe the cognitive speed bound and find the alpha knee.",
        "insights": [
            "JSON summaries from single runs (self-tuning optional).",
            "Alpha scans to spot throughput vs stability trade-offs.",
            "Control window and k_vert effects on the stack.",
        ],
        "data": [
            "Alpha, layer count, sites per layer, dt/steps, logging cadence.",
            "Alpha lists for scans.",
        ],
        "cases": [
            "Tune layered networks for latency/throughput limits.",
            "Stress-test hierarchical control or metabolic stacks.",
            "Benchmark candidate CLC settings before deployment.",
        ],
    },
}


def explainer_component(page_id: str) -> Optional[html.Div]:
    info = EXPLAINERS.get(page_id)
    if not info:
        return None

    sections: List[html.Div] = []
    for label, key in [
        ("Insights you can pull", "insights"),
        ("Data it works with", "data"),
        ("Example business cases", "cases"),
    ]:
        items = info.get(key) or []
        if not items:
            continue
        sections.append(
            html.Div(
                [html.Strong(label), html.Ul([html.Li(item) for item in items])],
                className="explainer-section",
            )
        )

    hint = info.get(
        "hint",
        "Tip: use Ask the Suite (Home) with your goal and industry; it will pre-fill the right flows here.",
    )

    return html.Details(
        [
            html.Summary(info.get("title", "What this tab does")),
            html.Div(info.get("purpose", ""), className="explainer-lede"),
            html.Div(sections, className="explainer-grid"),
            html.Div(hint, className="explainer-hint"),
        ],
        className="explainer",
        open=bool(info.get("open", True)),
    )

