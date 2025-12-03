# Adding a New UI Page to SACP Suite (Dash)

This guide shows how to plug a new page into the modular Dash shell. The goal: a self-contained page module that calls backend APIs (no heavy numerics in UI) and won’t break the app if it fails.

## Essentials
- Pages live under `src/sacp_suite/ui/pages/`.
- Each page module provides:
  - `layout()` → returns Dash components.
  - `register_callbacks(app)` → attaches callbacks to the provided Dash `app`.
  - A `register_page(...)` call with `id`, `name`, `path`, `layout`, `register_callbacks`, and an `order` for nav.
- The shell (`ui/app.py`) uses the registry to render and wire callbacks. Component IDs must be unique across the app.

## Shared helpers
Use `src/sacp_suite/ui/pages/common.py`:
- `API_BASE`: base URL for API calls (built from `SACP_BIND`/`SACP_PORT`).
- `get_shared_traj()` / `set_shared_traj(traj)`: in-process shared trajectory cache across pages.
- Parsers/coercers: `parse_numeric_list`, `parse_label_list`, `parse_patterns_text`, `parse_upload`, `coerce_int`, `coerce_float`.

## Page module template
```python
from dash import dcc, html, Input, Output, State, no_update
import requests
from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE, get_shared_traj, set_shared_traj  # as needed

API = API_BASE

def layout():
    return html.Div(
        [
            html.H3("My Module"),
            # controls…
            # graphs…
        ],
        id="my-module-root",
    )

def register_callbacks(app):
    @app.callback(
        Output("my-output", "children"),
        Input("my-button", "n_clicks"),
        State("my-input", "value"),
        prevent_initial_call=True,
    )
    def _do_work(n_clicks, value):
        if not n_clicks:
            raise PreventUpdate
        resp = requests.post(f"{API}/api/v1/my-module/endpoint", json={"param": value}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return f"Result: {data}"
    # add more callbacks as needed

register_page(
    "my-module-id",
    name="My Module",
    path="/my-module-path",
    layout=layout,
    register_callbacks=register_callbacks,
    order=120,  # nav order
)
```

## Backend router pattern (if adding endpoints)
- Place a new router under `src/sacp_suite/api/v1/<module>.py`.
- Include it in `api/main.py` with `app.include_router(<router>, prefix="/api/v1")`.
- Use Pydantic models for request/response; raise `HTTPException` for errors; lazy-import heavy modules to avoid startup failures.

## Styling
- Global styles: `src/sacp_suite/ui/assets/style.css`.
  - Navbar: `.sacp-navbar`, `.nav-link`.
  - Cards/buttons/layout: `.card`, `.section-title`, `.primary-btn`, `.upload-box`, `.mapping-grid`, etc.
- Reuse existing classes to stay consistent.

## Navigation and IDs
- `path` in `register_page` sets the URL (e.g., `/chemistry`).
- Use descriptive, page-unique component IDs to avoid clashes.
- Nav order controlled by `order` in `register_page`.

## API calls and wiring
- Use `requests` for backend calls; build URLs off `API_BASE` (or a module-specific env var if needed).
- Validate optional fields and handle failures; return user-friendly messages with `PreventUpdate`/`no_update` when appropriate.
- Do not import heavy numerics in the UI; keep the UI thin.

## Shared stores
- The shell injects `dcc.Store` components: `shared_traj`, `task_spec_store`, `landing_stage`, `ingestion_preview_store`, `dataset_builder_store`.
- For trajectories, prefer `get_shared_traj`/`set_shared_traj` from `common.py`.

## Chemistry example (pattern to follow)
- UI (`ui/pages/chemistry.py`) calls `/api/v1/chemistry/simulate` via `requests`, plots results; no direct numerics imports.
- Backend (`api/v1/chemistry.py`) isolates heavy imports and returns time-series + metadata.

## Testing/validation
- Syntax check: `python -m py_compile ...`.
- Run API: `python -m sacp_suite.api.main` (or `uvicorn sacp_suite.api.main:app`).
- Run UI: `python -m sacp_suite.ui.app` (or `sacp-suite` via console script).
- Visit `http://127.0.0.1:8050` and verify the new page renders/callbacks work.

## Guardrails
- Keep IDs stable/unique.
- Handle backend failures gracefully; don’t let exceptions bubble to Dash.
- Keep UI free of heavy imports; call FastAPI endpoints instead.
- Stick to the established layout/callback patterns for consistency.
