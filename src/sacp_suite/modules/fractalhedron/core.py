"""FractalHedron helpers plus a Lorenz/Attractorhedron integration sketch."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Symbolic coding + k-gram helpers
# ---------------------------------------------------------------------------

def build_symbolic_sequence(
    section_hits: np.ndarray,
    coding_spec: str = "x_sign",
    coding_params: Dict[str, object] | None = None,
) -> List[str]:
    """Convert section hits (e.g., Lorenz Poincaré intersections) into symbols."""

    data = np.asarray(section_hits, dtype=float)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError("section_hits must be shape (T, d>=1)")

    params = coding_params or {}
    x = data[:, 0]
    if coding_spec == "x_sign":
        return ["L" if xi < 0 else "R" for xi in x]
    if coding_spec == "quadrant_xz":
        if data.shape[1] < 3:
            raise ValueError("quadrant_xz coding requires at least 3 columns (x,y,z).")
        z = data[:, 2]
        out: List[str] = []
        for xi, zi in zip(x, z, strict=False):
            if xi >= 0 and zi >= 0:
                out.append("Q1")
            elif xi < 0 <= zi:
                out.append("Q2")
            elif xi < 0 and zi < 0:
                out.append("Q3")
            else:
                out.append("Q4")
        return out
    if coding_spec == "radius_bins":
        if data.shape[1] < 3:
            raise ValueError("radius_bins coding requires at least 3 columns (x,y,z).")
        z = data[:, 2]
        radius = np.sqrt(x**2 + z**2)
        bins = params.get("bins")
        if bins is None:
            r_min = float(radius.min())
            r_max = float(radius.max())
            if r_max - r_min < 1e-8:
                bins = [r_min, r_min + 1e-3, r_min + 2e-3, float("inf")]
            else:
                step = (r_max - r_min) / 3.0
                bins = [r_min, r_min + step, r_min + 2 * step, float("inf")]
        labels = params.get("labels", ["inner", "middle", "outer"])
        if len(labels) != len(bins) - 1:
            raise ValueError("labels length must match len(bins) - 1 for radius_bins coding.")
        indices = np.digitize(radius, bins[1:-1], right=False)
        return [labels[idx] for idx in indices]
    raise NotImplementedError(f"Unknown coding_spec: {coding_spec}")


def build_kgram_counts(symbolic_seq: Sequence[str], k: int) -> Counter:
    """Count k-grams inside a discrete symbolic sequence."""

    if k < 1:
        raise ValueError("k must be >= 1")
    counts: Counter = Counter()
    n = len(symbolic_seq)
    if n < k:
        return counts
    for i in range(n - k + 1):
        w = tuple(symbolic_seq[i : i + k])
        counts[w] += 1
    return counts


def normalize_kgram_probs(kgram_counts: Counter) -> Dict[Tuple[str, ...], float]:
    """Normalize counts into probabilities, guarding against empty sequences."""

    total = float(sum(kgram_counts.values()))
    if total <= 0:
        return {word: 0.0 for word in kgram_counts.keys()}
    return {word: count / total for word, count in kgram_counts.items()}


# ---------------------------------------------------------------------------
# Multifractal moments + polymorphic scale model
# ---------------------------------------------------------------------------

def default_scale_model(alphabet_size: int, k: int) -> float:
    """Symbolic scale ℓₖ = |A|^{-k}."""

    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    return 1.0 / (alphabet_size**k)


def _infer_alphabet(words: Iterable[Tuple[str, ...]]) -> List[str]:
    alphabet = set()
    for word in words:
        alphabet.update(word)
    return sorted(alphabet)


def fractal_moments(
    kgram_probs: Dict[Tuple[str, ...], float],
    *,
    k: int,
    Q: Sequence[float],
    scale_model=default_scale_model,
    alphabet_size: int | None = None,
) -> Dict[str, Dict[float, float] | float | bool]:
    """Compute symbolic multifractal moments."""

    if alphabet_size is None:
        alphabet_size = len(_infer_alphabet(kgram_probs.keys()))
    if alphabet_size == 0:
        return {
            "q_moments": {q: {"T_q": 0.0, "D_q": 0.0} for q in Q},
            "D_values": {q: 0.0 for q in Q},
            "ell_k": 0.0,
            "alphabet_size": 0,
            "monotone_ok": True,
            "bounds_ok": True,
        }

    ell_k = scale_model(alphabet_size, k)
    if not (0 < ell_k < 1):
        raise ValueError(f"scale_model must return ℓₖ in (0,1); got {ell_k}")

    log_inv_ellk = log(1.0 / ell_k)
    q_moments: Dict[float, Dict[str, float]] = {}
    D_values: Dict[float, float] = {}
    for q in Q:
        if q == 1.0:
            raise ValueError("q=1 not implemented; use q→1 limit externally.")
        T_q = sum(p**q for p in kgram_probs.values())
        D_q = 0.0 if T_q <= 0.0 else (1.0 / (q - 1.0)) * (log(T_q) / log_inv_ellk)
        q_moments[q] = {"T_q": T_q, "D_q": D_q}
        D_values[q] = D_q

    q_sorted = sorted(Q)
    monotone_ok = all(D_values[q_sorted[i]] >= D_values[q_sorted[i + 1]] - 1e-9 for i in range(len(q_sorted) - 1))
    bounds_eps = 5e-2
    bounds_ok = all(-bounds_eps <= D <= 1.0 + bounds_eps for D in D_values.values())

    return {
        "q_moments": q_moments,
        "D_values": D_values,
        "ell_k": ell_k,
        "alphabet_size": alphabet_size,
        "monotone_ok": monotone_ok,
        "bounds_ok": bounds_ok,
    }


def build_fractalhedron_k(
    symbolic_seq: Sequence[str],
    *,
    k: int,
    Q: Sequence[float],
    scale_model=default_scale_model,
) -> Dict[str, object]:
    """Bundle symbolic coding + multifractal stats into a single dict."""

    kgram_counts = build_kgram_counts(symbolic_seq, k)
    probs = normalize_kgram_probs(kgram_counts)
    fm = fractal_moments(
        probs,
        k=k,
        Q=Q,
        scale_model=scale_model,
        alphabet_size=None,
    )
    alphabet = _infer_alphabet(probs.keys())
    return {
        "k": k,
        "alphabet": alphabet,
        "p": probs,
        "D_q": fm["D_values"],
        "T_q": {q: fm["q_moments"][q]["T_q"] for q in Q},
        "ell_k": fm["ell_k"],
        "constraints": {
            "monotone_ok": bool(fm["monotone_ok"]),
            "bounds_ok": bool(fm["bounds_ok"]),
            "D_max": 1.0,
        },
    }


def fractal_face_flags(fractalhedron_k: Dict[str, object], *, eps_p: float = 1e-4, eps_D: float = 1e-2) -> Dict[str, List]:
    """Detect faces (zero-probability words, monofractal bands, extremal D_q)."""

    probs: Dict[Tuple[str, ...], float] = fractalhedron_k.get("p", {})
    D_q: Dict[float, float] = fractalhedron_k.get("D_q", {})
    D_max = fractalhedron_k.get("constraints", {}).get("D_max", 1.0)

    zero_words = [word for word, prob in probs.items() if prob < eps_p]
    q_list = sorted(D_q.keys())
    monofractal_pairs = [
        (q1, q2)
        for idx, q1 in enumerate(q_list)
        for q2 in q_list[idx + 1 :]
        if abs(D_q[q1] - D_q[q2]) < eps_D
    ]
    near_max_dim = [q for q, D in D_q.items() if D > D_max - eps_D]
    near_zero_dim = [q for q, D in D_q.items() if D < eps_D]
    return {
        "symbolic_zero_words": zero_words,
        "monofractal_pairs": monofractal_pairs,
        "near_max_dim": near_max_dim,
        "near_zero_dim": near_zero_dim,
    }


# ---------------------------------------------------------------------------
# Lorenz/Attractorhedron integration sketch
# ---------------------------------------------------------------------------

def run_fractalhedron2_example(
    section_hits: np.ndarray,
    *,
    coding_spec: str = "x_sign",
    k: int = 2,
    q_values: Sequence[float] = (0.0, 2.0),
) -> Tuple[Dict[str, object], Dict[str, List]]:
    """Convenience helper that mirrors the README-ready v1 script."""

    sym_seq = build_symbolic_sequence(section_hits, coding_spec=coding_spec)
    fh2 = build_fractalhedron_k(sym_seq, k=k, Q=q_values)
    faces = fractal_face_flags(fh2, eps_p=1e-5, eps_D=1e-2)
    return fh2, faces


# ---------------------------------------------------------------------------
# Agenthedron hook (fractal penalty term)
# ---------------------------------------------------------------------------

@dataclass
class AgentCostTerms:
    """Small container for the Agenthedron cost breakdown."""

    FFE: float
    FFIE: float
    Yield: float
    Risk: float


def compute_fractal_penalty(
    fractalhedron_k: Dict[str, object],
    *,
    target_D2: float | None = None,
    weight: float = 1.0,
) -> float:
    """Squared penalty pushing D₂ towards a target (defaults to D_max)."""

    D_q = fractalhedron_k.get("D_q", {})
    D2 = D_q.get(2.0)
    if D2 is None:
        return 0.0
    if target_D2 is None:
        target_D2 = fractalhedron_k.get("constraints", {}).get("D_max", 1.0)
    return weight * float(D2 - target_D2) ** 2


def agent_cost_with_fractal(
    terms: AgentCostTerms,
    fhk: Dict[str, object],
    *,
    weight_ffe: float = 1.0,
    weight_ffie: float = 1.0,
    weight_yield: float = 1.0,
    weight_risk: float = 1.0,
    weight_fractal: float = 1.0,
    target_D2: float | None = None,
) -> float:
    """Agenthedron J = aFFE + bFFIE − cYield + dRisk + fractal penalty."""

    base = (
        weight_ffe * terms.FFE
        + weight_ffie * terms.FFIE
        - weight_yield * terms.Yield
        + weight_risk * terms.Risk
    )
    penalty = compute_fractal_penalty(fhk, target_D2=target_D2, weight=weight_fractal)
    return float(base + penalty)
