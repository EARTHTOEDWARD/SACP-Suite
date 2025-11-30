import pytest
from fastapi.testclient import TestClient

from sacp_suite.api.main import app


client = TestClient(app)


def test_cog_memory_endpoint_basic():
    payload = {
        "n_trials": 2,
        "n_steps": 400,
        "burn_in": 50,
        "max_lag": 15,
        "n_output_bins": 8,
        "amp": 1.5,
        "dt": 0.02,
        "input_alphabet": [-1.0, 0.0, 1.0],
    }
    resp = client.post("/cog/memory", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["lags"]) == payload["max_lag"]
    assert len(data["M"]) == payload["max_lag"]
    assert len(data["sample_u"]) == payload["n_steps"]
    assert len(data["sample_y"]) == payload["n_steps"]
    assert pytest.approx(data["dt"], rel=1e-6) == payload["dt"]
    assert all(v >= 0.0 for v in data["M"])


def test_cog_discriminability_endpoint_basic():
    payload = {
        "patterns": [
            {"pattern": [1.0, 0.0, -1.0, 0.0]},
            {"pattern": [1.0, 1.0, 0.0, -1.0]},
        ],
        "n_trials_per_concept": 8,
        "burn_in": 60,
        "amp": 1.0,
        "dt": 0.02,
    }
    resp = client.post("/cog/discriminability", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["K"] == len(payload["patterns"])
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["D"] <= 1.0
    assert data["T"] >= len(payload["patterns"][0]["pattern"])
