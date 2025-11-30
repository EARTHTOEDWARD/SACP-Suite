# Self-Tuning Adapter

This directory vendors the `selftuner` helpers into the SACP repo and wires them to SACP systems via a small adapter layer.

```
src/sacp_suite/
├── selftuner/          # generic helpers (Lyapunov, controller, fd tangent…)
└── self_tuning/       # SACP glue (protocol + manager)
```

To expose a system to the tuner:

1. Implement `SelfTunableSystem` (see `self_tuning/protocols.py`). Existing systems only need to provide `id`, `dt`, `get_state_vector()`, and `apply_tuning_action()`.
2. Construct `SelfTuningManager` with the system, plus either a `system_step_fn` (pure step for FD tangent) or a custom `tangent_map`.
3. Call `manager.update_for_current_state()` after each normal simulation step. The manager keeps a streaming Lyapunov estimate and pushes tuning actions back into the system.
4. Surface diagnostics via `manager.state.last_estimate` / `manager.state.last_action` or helper getters for UI work.

Controller defaults keep the largest Lyapunov exponent inside a configurable band. Adjust `ChaosBand`, `k_spectral_radius`, and `k_gain` to suit each modality.
