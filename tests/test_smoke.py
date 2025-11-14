import numpy as np

from sacp_suite.core.plugin_api import registry
from sacp_suite.modules.sacp_x import lorenz63  # noqa: F401 ensures plugin registration
from sacp_suite.modules.attractorhedron.operator import build_ulam, analyze_operator


def test_simulate_basic():
    dyn = registry.create_dynamics("lorenz63", rho=28.0)
    traj = dyn.simulate(T=1.0, dt=0.01)
    assert traj.shape[0] > 10
    assert traj.shape[1] == 3


def test_operator_pipeline():
    t = np.linspace(0, 2 * np.pi, 1000)
    pts = np.stack([np.cos(t), np.sin(t)], axis=1)
    P, _ = build_ulam(pts, nx=10, ny=10)
    res = analyze_operator(P, mean_return=1.0, nx=10, ny=10)
    assert 0 <= res["abs_lambda2"] <= 1
