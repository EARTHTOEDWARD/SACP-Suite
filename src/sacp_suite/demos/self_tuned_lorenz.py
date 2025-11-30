"""Self-tuned Lorenz attractor demo script."""

from __future__ import annotations

from typing import Dict, List

from sacp_suite.self_tuning.manager import SelfTuningManager
from sacp_suite.selftuner.types import ChaosBand
from sacp_suite.systems.lorenz import LorenzSystem


def run_self_tuned_lorenz(
    num_steps: int = 10_000,
    print_every: int = 1_000,
    record_states: bool = False,
) -> Dict[str, List]:
    system = LorenzSystem(id_="lorenz_demo", dt=0.01)
    manager = SelfTuningManager(
        system=system,
        state_dim=system.get_state_vector().shape[0],
        system_step_fn=system.step_pure,
        chaos_band=ChaosBand(lower=0.0, upper=0.5),
        n_exponents=1,
        k_spectral_radius=0.05,
        modality="lorenz",
        extras={"demo": True},
    )

    lambdas: List[float] = []
    regimes: List[str] = []
    spectral_radius: List[float] = []
    gains: List[float] = []
    states: List[List[float]] = []

    for idx in range(num_steps):
        system.step()
        manager.update_for_current_state()
        lam = manager.get_latest_lambda_max()
        regime = manager.get_latest_regime()
        if lam is not None:
            lambdas.append(lam)
            regimes.append(regime or "")
        spectral_radius.append(system.spectral_radius)
        gains.append(system.gain)
        if record_states:
            states.append(system.get_state_vector().tolist())
        if lam is not None and (idx + 1) % print_every == 0:
            print(
                f"step {idx + 1:6d} | "
                f"lambda_max = {lam: .5f} | "
                f"regime={regime or 'n/a':>12} | "
                f"rho_scale = {system.spectral_radius: .3f} | "
                f"gain = {system.gain: .3f}"
            )
    return {
        "lambdas": lambdas,
        "regimes": regimes,
        "spectral_radius": spectral_radius,
        "gains": gains,
        "states": states,
        "dt": system.dt,
    }


if __name__ == "__main__":
    run_self_tuned_lorenz()
