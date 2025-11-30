import numpy as np

from fastapi.testclient import TestClient

from sacp_suite.api.main import app
client = TestClient(app)

from sacp_suite.modules.fractalhedron.core import (
    AgentCostTerms,
    agent_cost_with_fractal,
    build_fractalhedron_k,
    build_symbolic_sequence,
    run_fractalhedron2_example,
)


def _sample_section_hits(n: int = 200):
    t = np.linspace(0, 4 * np.pi, n)
    x = np.sin(t)
    y = np.cos(t)
    z = np.sin(2 * t)
    return np.stack([x, y, z], axis=1)


def test_build_symbolic_sequence_binary_code():
    hits = _sample_section_hits(50)
    symbols = build_symbolic_sequence(hits, coding_spec="x_sign")
    assert len(symbols) == hits.shape[0]
    assert {"L", "R"} <= set(symbols)


def test_build_symbolic_sequence_quadrant():
    hits = _sample_section_hits(40)
    symbols = build_symbolic_sequence(hits, coding_spec="quadrant_xz")
    assert len(symbols) == hits.shape[0]
    assert set(symbols) <= {"Q1", "Q2", "Q3", "Q4"}


def test_build_symbolic_sequence_radius_bins():
    hits = _sample_section_hits(40)
    symbols = build_symbolic_sequence(
        hits,
        coding_spec="radius_bins",
        coding_params={"bins": [0.0, 0.3, 0.6, float("inf")], "labels": ["low", "mid", "high"]},
    )
    assert len(symbols) == hits.shape[0]
    assert set(symbols) <= {"low", "mid", "high"}


def test_build_fractalhedron_k_simple_sequence():
    seq = ["L", "R", "L", "R", "L"]
    fh2 = build_fractalhedron_k(seq, k=2, Q=(0.0, 2.0))
    assert fh2["k"] == 2
    assert fh2["alphabet"] == ["L", "R"]
    assert set(fh2["D_q"].keys()) == {0.0, 2.0}
    assert "bounds_ok" in fh2["constraints"]


def test_run_fractalhedron2_example_end_to_end():
    hits = _sample_section_hits(300)
    fh2, faces = run_fractalhedron2_example(hits)
    assert fh2["alphabet"] == ["L", "R"]
    assert "symbolic_zero_words" in faces
    assert isinstance(faces["monofractal_pairs"], list)


def test_agent_cost_with_fractal_penalty():
    seq = ["L", "R", "L", "R", "L", "R"]
    fh2 = build_fractalhedron_k(seq, k=2, Q=(0.0, 2.0))
    terms = AgentCostTerms(FFE=0.5, FFIE=0.4, Yield=0.2, Risk=0.1)
    cost = agent_cost_with_fractal(terms, fh2, weight_fractal=1.5, target_D2=0.9)
    base = 0.5 + 0.4 - 0.2 + 0.1
    assert cost >= base


def test_fractalhedron_api_endpoint():
    hits = _sample_section_hits(120)
    payload = {
        "section_hits": hits.tolist(),
        "coding_spec": "quadrant_xz",
        "k": 2,
        "Q": [0.0, 2.0],
    }
    resp = client.post("/fractalhedron/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["alphabet"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert "D_q" in data and "faces" in data


def test_fractalhedron_api_endpoint_uses_cached_hits():
    hits = _sample_section_hits(90)
    # First call with hits to populate cache
    client.post(
        "/fractalhedron/run",
        json={"section_hits": hits.tolist(), "coding_spec": "x_sign", "k": 2, "Q": [0.0, 2.0]},
    )
    # Second call without hits should use cache
    resp = client.post("/fractalhedron/run", json={"section_hits": [], "coding_spec": "x_sign"})
    assert resp.status_code == 200
