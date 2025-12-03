# SACP Suite Codebase Overview

SACP Suite is a unified platform that bundles several chaos-analysis tools into one monorepo. It exposes a FastAPI backend, a Dash-based UI, and a plugin-friendly core so new dynamical systems or analytics can be dropped in without touching shared infrastructure. This document maps the architecture, main modules, key entry points, and operational details so you can quickly author new analyses or integrations.

## Repository Layout

- `src/sacp_suite/api/` – FastAPI service that surfaces simulations, metrics, operator analysis, gating, and health checks.
- `src/sacp_suite/ui/` – Dash multi-page web UI that wraps the API for interactive simulation and attractor inspection.
- `src/sacp_suite/core/` – Plugin definitions and registries used by every subsystem.
- `src/sacp_suite/modules/` – Feature modules:
  - `sacp_x/` – Lorenz 63 dynamics plus chaos metrics.
  - `attractorhedron/` – Ulam operator construction, spectral analysis, and gating utilities.
  - `bcp/` – Bioelectric Control Panel I/O scaffolding.
  - `abtc/` – Attractor-Based Trajectory Calculator stubs and shared integrators.
  - `maes/` – Reserved for future expansions.
- `tests/` – Smoke tests for simulations and operator analysis.
- Root files: `pyproject.toml`, `Makefile`, `.env.example`.

All components live in a single package/version; modules under `modules/` auto-register themselves so the suite can be extended by simply adding new directories.

## Launching and Entry Points

Console scripts (declared in `pyproject.toml`) are the standard entry points:

- `sacp-suite-api` → runs `sacp_suite.api.main:run()` via Uvicorn and exposes FastAPI (defaults to `127.0.0.1:8000`).
- `sacp-suite-ui` → runs `sacp_suite.ui.app:main()` and serves the Dash UI (defaults to `127.0.0.1:8050`).

For development there are `make api` and `make ui` helpers to launch API and UI together. The API binds to localhost unless `SACP_BIND=0.0.0.0` is set. CORS is disabled until `SACP_CORS` lists allowed origins. FastAPI ships Swagger docs at `http://127.0.0.1:8000/docs`; Dash lives at `http://127.0.0.1:8050`.

## Core Plugin Framework (`sacp_suite/core/plugin_api.py`)

**BaseDynamics**

- Abstract interface for every dynamical system plugin; requires `get_default_params()`, `default_state()`, `derivative(state, t)`, `name`, and `state_labels`.
- Ships with `simulate(T, dt, x0)` that runs a 4th-order Runge-Kutta integrator (`rk4_step`) and returns an `(steps, state_dim)` trajectory.

**PluginRegistry**

- Registers and instantiates dynamics classes through `register_dynamics(key, cls)`, `list_dynamics()`, and `create_dynamics(key, **params)`.
- A module-level singleton (`registry`) powers API discovery and instantiation. Importing a module that registers a plugin (e.g., Lorenz) makes it globally available.

## Module Highlights

### SACP-X (`sacp_suite/modules/sacp_x`)

- `lorenz63.py`: Implements `Lorenz63(BaseDynamics)` with parameters `sigma`, `rho`, `beta`, default state `(1, 1, 1)`, labeled axes, and registry hookup via `registry.register_dynamics("lorenz63", Lorenz63)`.
- `metrics.py`: Provides Rosenstein Largest Lyapunov Exponent via `rosenstein_lle(series, m=6, tau=4)`; performs delay embedding, neighbor tracking, and slope fitting to estimate λ₁. API endpoint `/metrics/lle` wraps this helper.

### Attractorhedron (`sacp_suite/modules/attractorhedron`)

- `operator.py`:
  - `build_ulam(points, nx=25, ny=25, bounds=None)`: Grids a 2D projection, counts transitions, and returns a row-stochastic matrix `P` plus quantile bounds.
  - `analyze_operator(P, mean_return=1.0, nx=25, ny=25)`: Computes eigenpairs of `Pᵀ`, surfaces `lambda2`, `abs_lambda2`, `gamma=-ln(|λ₂|)/mean_return`, and reshapes the second eigenvector into `ny×nx` for visualization.
  - `left_right_gate(P, nx, ny, alpha=1.0)`: Scales cross-half transitions, renormalizes rows, and models gating perturbations; `/gate/sweep` applies multiple `alpha` values.

### BCP (`sacp_suite/modules/bcp`)

- `interface.py` defines a `Recording` dataclass (`id`, timestamps `t`, `channels`, `values`) as the canonical container for uploaded multichannel data.
- `to_section(points)` stub returns the first two columns to mimic a Poincaré section placeholder; future work will implement real slicing/down-sampling.

### ABTC (`sacp_suite/modules/abtc`)

- `core.py` exposes an RK4 integrator `rk4(f, x0, dt, n)` useful for upcoming trajectory planning/control solvers.

## Dash UI (`sacp_suite/ui/app.py`)

- Multipage layout with shared navigation.
- **Home**: Welcome text plus links and API endpoint info.
- **Simulator**: Controls for Lorenz parameters (`rho`, `sigma`, `beta`, `T`, `dt`), runs `/simulate`, plots Plotly 3D phase portrait, per-axis time series, and computes LLE via `/metrics/lle`. Stores the most recent trajectory on the server for reuse.
- **Attractorhedron**: Loads the stored trajectory, builds `/operator/build` from (X, Z) coordinates, analyzes it via `/operator/analyze`, displays `|λ₂|`, `gamma`, and renders the `v2_field` heatmap.
- HTTP calls use `requests`, keeping UI logic lightweight and consistent with external API consumers.

## API Surface (`sacp_suite/api/main.py`)

- `GET /health` → `{ "ok": true, "models": [...] }`.
- `POST /simulate` → takes `model`, `params`, `T`, `dt`; returns `time`, `trajectory`, `labels`.
- `POST /metrics/lle` → accepts `series`, optional `m`, `tau`; returns `{ "lle": value }`.
- `POST /operator/build` → ingests `points`, `nx`, `ny`; returns `{ "P": matrix, "bounds": [xmin, xmax, zmin, zmax] }`.
- `POST /operator/analyze` → consumes `P`, `mean_return`, `nx`, `ny`; returns spectral stats and `v2_field`.
- `POST /gate/sweep` → sweeps `alpha` values through `left_right_gate`, reuses analyze output per case.

All payloads are validated through Pydantic models, and numerical arrays are converted to lists for JSON transport.

## Configuration & Deployment

Environment variables (see `.env.example`) govern runtime behavior:

- `SACP_BIND` / `SACP_PORT` – FastAPI host/port (defaults `127.0.0.1:8000`).
- `SACP_UI_HOST` / `SACP_UI_PORT` – Dash host/port (defaults `127.0.0.1:8050`).
- `SACP_CORS` – Comma-separated origins to enable CORS.
- `SACP_DATA_DIR` – Upload directory (default `var/uploads`, `.gitignore`d).

Both FastAPI and Dash read these variables during startup, allowing identical configuration for local, container, or cloud deployments.

## Dependencies

Declared in `pyproject.toml` with optional extras (`api`, `ui`, `all`):

- FastAPI, Uvicorn – backend service.
- Dash, Plotly – UI stack.
- NumPy, SciPy, pandas, scikit-learn – numerical backbone.
- requests – Dash-to-API communication.
- python-multipart – enables future file uploads.

Install everything via `pip install "sacp-suite[all]"`; Makefile targets wrap common dev workflows (`make format`, `make test`, etc.).

## Testing and Extensibility

- `tests/test_smoke.py` exercises Lorenz simulation (shape checks) and Ulam operator creation/analysis (ensures spectral outputs within valid ranges). Run with `pytest` or `make test`.
- To add a new dynamical system: subclass `BaseDynamics`, implement required hooks, call `registry.register_dynamics("key", YourClass)`, and import the module in API startup if autoloading is needed.
- To add analytics or metrics: create helper functions (similar to `rosenstein_lle`), expose them via new FastAPI endpoints, and optionally surface them in Dash through new callbacks or pages.
- To ingest real data: extend BCP to load files into `Recording`, add upload endpoints that save to `SACP_DATA_DIR`, and feed those recordings into operator/metric routines.

The shared plugin registry, JSON API, and UI callbacks provide clear extension seams for new systems, controllers, or visualizations without refactoring existing code.
