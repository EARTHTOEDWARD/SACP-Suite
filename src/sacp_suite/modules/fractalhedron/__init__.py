"""Symbolic multifractal helpers (FractalHedron_k)."""

from .core import (
    AgentCostTerms,
    agent_cost_with_fractal,
    build_fractalhedron_k,
    build_kgram_counts,
    build_symbolic_sequence,
    compute_fractal_penalty,
    default_scale_model,
    fractal_face_flags,
    fractal_moments,
    normalize_kgram_probs,
    run_fractalhedron2_example,
)

__all__ = [
    "AgentCostTerms",
    "agent_cost_with_fractal",
    "build_fractalhedron_k",
    "build_kgram_counts",
    "build_symbolic_sequence",
    "compute_fractal_penalty",
    "default_scale_model",
    "fractal_face_flags",
    "fractal_moments",
    "normalize_kgram_probs",
    "run_fractalhedron2_example",
]
