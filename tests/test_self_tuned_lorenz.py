from fastapi.testclient import TestClient

from sacp_suite.api.main import app
from sacp_suite.demos.self_tuned_lorenz import run_self_tuned_lorenz
from sacp_suite.systems.lorenz import LorenzSystem

client = TestClient(app)


def test_lorenz_system_step_changes_state():
    system = LorenzSystem()
    before = system.get_state_vector().copy()
    system.step()
    after = system.get_state_vector()
    assert not (before == after).all()


def test_self_tuned_lorenz_demo_runs():
    result = run_self_tuned_lorenz(num_steps=20, print_every=10, record_states=True)
    assert len(result["lambdas"]) > 0
    assert len(result["lambdas"]) == len(result["regimes"])
    assert len(result["spectral_radius"]) == 20
    assert len(result["states"]) == 20


def test_self_tuning_lorenz_api():
    resp = client.post("/self-tuning/lorenz", json={"num_steps": 30, "record_states": False})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["lambdas"]) > 0


def test_self_tuning_session_flow():
    create = client.post("/self-tuning/sessions", json={"dt": 0.01})
    assert create.status_code == 200
    session_id = create.json()["session_id"]
    step_resp = client.post("/self-tuning/sessions/step", json={"session_id": session_id, "steps": 10})
    assert step_resp.status_code == 200
    data = step_resp.json()
    assert data["step"] == 10
    assert len(data["lambdas"]) > 0
