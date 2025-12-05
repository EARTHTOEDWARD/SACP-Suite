from __future__ import annotations

from typing import Any, Dict, List

import dash
import numpy as np
import plotly.graph_objects as go
import requests
from dash import ALL, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE, set_shared_traj, PLOTLY_DARK_LAYOUT

API = API_BASE


def layout():
    return html.Div(
        [
            dcc.Store(id="sa_meta_store"),
            dcc.Interval(id="sa_init_tick", interval=500, n_intervals=0, max_intervals=1),
            html.Div(
                [
                    html.H3("Strange Attractors Library", className="section-title"),
                    html.P(
                        "Pick an attractor from the library inspired by the Shashank Tomar gallery, tweak parameters, and overlay multiple seeds.",
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Attractor"),
                            dcc.Dropdown(id="sa_model", clearable=False),
                            html.Div(id="sa_meta_status", className="status-text", style={"marginTop": "4px"}),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(id="sa_param_controls", style={"marginTop": "10px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Duration T"),
                                    dcc.Slider(id="sa_T", min=10, max=120, step=5, value=60, tooltip={"placement": "bottom"}),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("dt"),
                                    dcc.Slider(
                                        id="sa_dt",
                                        min=0.001,
                                        max=0.05,
                                        step=0.001,
                                        value=0.01,
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Seeds / overlays"),
                                    dcc.Slider(
                                        id="sa_seeds",
                                        min=1,
                                        max=6,
                                        step=1,
                                        value=2,
                                        marks={1: "1", 3: "3", 6: "6"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Seed jitter (σ)"),
                                    dcc.Slider(
                                        id="sa_jitter",
                                        min=0.0,
                                        max=0.2,
                                        step=0.01,
                                        value=0.05,
                                        marks={0.0: "0", 0.1: "0.1", 0.2: "0.2"},
                                    ),
                                ]
                            ),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run attractor", id="sa_run", n_clicks=0, className="primary-btn", style={"marginTop": "8px"}),
                    html.Div(id="sa_run_status", className="status-text", style={"marginTop": "6px"}),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("Phase portrait (3D) and time series", className="section-title"),
                    html.Div(
                        [
                            dcc.Loading(dcc.Graph(id="sa_phase3d"), type="dot"),
                            dcc.Loading(dcc.Graph(id="sa_series"), type="dot"),
                        ],
                        className="section-grid",
                    ),
                    html.Div(
                        "Tip: the first run is stored as shared trajectory; open Attractorhedron, Fractal LLM, or Simulator and click \"Show shared trajectory\".",
                        className="status-text",
                        style={"marginTop": "6px"},
                    ),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
            html.Div(
                [
                    html.H3("Composite (coupled) builder", className="section-title"),
                    html.P(
                        "Run two or more attractors with diffusive all-to-all coupling. Uses default parameters per attractor.",
                        style={"marginBottom": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Attractors to couple"),
                            dcc.Dropdown(id="sa_combo_models", multi=True, placeholder="Pick 2–4 attractors"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Coupling k"),
                                    dcc.Slider(
                                        id="sa_combo_k",
                                        min=0.0,
                                        max=0.2,
                                        step=0.01,
                                        value=0.05,
                                        marks={0.0: "0", 0.1: "0.1", 0.2: "0.2"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("dt"),
                                    dcc.Slider(
                                        id="sa_combo_dt",
                                        min=0.001,
                                        max=0.05,
                                        step=0.001,
                                        value=0.01,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Duration T"),
                                    dcc.Slider(
                                        id="sa_combo_T",
                                        min=10,
                                        max=120,
                                        step=5,
                                        value=50,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Initial jitter (σ)"),
                                    dcc.Slider(
                                        id="sa_combo_jitter",
                                        min=0.0,
                                        max=0.3,
                                        step=0.01,
                                        value=0.05,
                                        marks={0.0: "0", 0.15: "0.15", 0.3: "0.3"},
                                    ),
                                ]
                            ),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run coupled system", id="sa_combo_run", n_clicks=0, className="primary-btn", style={"marginTop": "8px"}),
                    html.Div(id="sa_combo_status", className="status-text", style={"marginTop": "6px"}),
                    html.Div(
                        [
                            dcc.Loading(dcc.Graph(id="sa_combo_phase3d"), type="dot"),
                            dcc.Loading(dcc.Graph(id="sa_combo_series"), type="dot"),
                        ],
                        className="section-grid",
                    ),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
        ]
    )


def _call_api_list() -> Dict[str, Any]:
    resp = requests.get(f"{API}/attractors", timeout=10)
    resp.raise_for_status()
    return resp.json()


def register_callbacks(app):
    @app.callback(
        Output("sa_meta_store", "data"),
        Output("sa_model", "options"),
        Output("sa_model", "value"),
        Output("sa_combo_models", "options"),
        Output("sa_meta_status", "children"),
        Input("sa_init_tick", "n_intervals"),
        prevent_initial_call=False,
    )
    def load_meta(n: int):
        try:
            data = _call_api_list()
            items = data.get("attractors", [])
            if not items:
                return dash.no_update, [], None, [], "No attractors available from API."
            options = [{"label": item["name"], "value": item["key"]} for item in items]
            return items, options, items[0]["key"], options, f"Loaded {len(items)} attractors."
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, [], None, [], f"Could not load library: {exc}"

    @app.callback(
        Output("sa_param_controls", "children"),
        Input("sa_model", "value"),
        State("sa_meta_store", "data"),
    )
    def render_params(model_key: str | None, meta: List[Dict[str, Any]] | None):
        if not model_key or not meta:
            return ""
        found = next((m for m in meta if m.get("key") == model_key), None)
        if not found:
            return html.Div("Unknown attractor selection.", className="status-text error")
        params = found.get("params", {})
        controls: List[Any] = []
        for name, spec in params.items():
            controls.append(
                html.Div(
                    [
                        html.Label(f"{name}"),
                        dcc.Slider(
                            id={"type": "sa-param", "param": name},
                            min=spec.get("min", 0.0),
                            max=spec.get("max", spec.get("default", 1.0) * 2),
                            step=spec.get("step", 0.01),
                            value=spec.get("default"),
                            tooltip={"placement": "bottom"},
                        ),
                    ]
                )
            )
        return html.Div(controls, className="mapping-grid")

    @app.callback(
        Output("sa_phase3d", "figure", allow_duplicate=True),
        Output("sa_series", "figure", allow_duplicate=True),
        Output("sa_run_status", "children", allow_duplicate=True),
        Output("shared_traj", "data", allow_duplicate=True),
        Input("sa_run", "n_clicks"),
        State("sa_model", "value"),
        State({"type": "sa-param", "param": ALL}, "id"),
        State({"type": "sa-param", "param": ALL}, "value"),
        State("sa_T", "value"),
        State("sa_dt", "value"),
        State("sa_seeds", "value"),
        State("sa_jitter", "value"),
        State("sa_meta_store", "data"),
        prevent_initial_call=True,
    )
    def run_attractor(
        n_clicks: int,
        model_key: str | None,
        param_ids: List[Dict[str, str]],
        param_values: List[float],
        T: float,
        dt: float,
        seeds: int,
        jitter: float,
        meta: List[Dict[str, Any]] | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not model_key:
            return dash.no_update, dash.no_update, "Pick an attractor first.", dash.no_update
        meta_map = {m["key"]: m for m in (meta or []) if "key" in m}
        base_state = np.array(meta_map.get(model_key, {}).get("default_state", [1.0, 1.0, 1.0]), dtype=float)
        params = {}
        for pid, val in zip(param_ids, param_values):
            if pid and "param" in pid:
                params[pid["param"]] = float(val)
        seeds = max(1, int(seeds or 1))
        jitter = float(jitter or 0.0)

        runs: List[np.ndarray] = []
        labels: List[str] = []
        statuses: List[str] = []
        for seed_idx in range(seeds):
            x0 = base_state.copy()
            if jitter > 0:
                x0 = x0 + np.random.normal(0.0, jitter, size=3)
            else:
                # tiny deterministic nudge so multiple runs don't collapse on identical trajectories
                x0 = x0 + 0.001 * (seed_idx + 1)
            try:
                req = {"model": model_key, "params": params, "T": T, "dt": dt, "x0": x0.tolist()}
                resp = requests.post(f"{API}/simulate", json=req, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                traj = np.array(data.get("trajectory", []), dtype=float)
                lab = data.get("labels", ["X", "Y", "Z"])
                if traj.ndim != 2 or traj.shape[0] == 0:
                    statuses.append(f"Seed {seed_idx+1}: empty trajectory.")
                    continue
                runs.append(traj)
                labels = lab
                statuses.append(f"Seed {seed_idx+1}: {traj.shape[0]} steps.")
            except Exception as exc:  # noqa: BLE001
                statuses.append(f"Seed {seed_idx+1} failed: {exc}")

        if not runs:
            return go.Figure(), go.Figure(), "; ".join(statuses), dash.no_update

        # Store first run for cross-page sharing.
        set_shared_traj(runs[0])

        fig3 = go.Figure()
        colors = ["#2563eb", "#10b981", "#f97316", "#6366f1", "#ef4444", "#14b8a6"]
        for idx, traj in enumerate(runs):
            fig3.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                    mode="lines",
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    name=f"Run {idx + 1}",
                )
            )
        fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_ts = go.Figure()
        time_axis = np.arange(runs[0].shape[0]) * dt
        names = labels or ["X", "Y", "Z"]
        for idx, traj in enumerate(runs):
            alpha = max(0.4, 1.0 - 0.12 * idx)
            for dim in range(min(3, traj.shape[1])):
                fig_ts.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=traj[:, dim],
                        mode="lines",
                        name=f"Run {idx + 1} · {names[dim]}",
                        line=dict(width=1.5, color=colors[idx % len(colors)]),
                        opacity=alpha,
                    )
                )
        fig_ts.update_layout(height=320, xaxis_title="t", legend=dict(orientation="h"))

        status = f"Ran {len(runs)} seed(s) for {T}s at dt={dt}. " + "; ".join(statuses)
        return fig3, fig_ts, status, runs[0].tolist()

    @app.callback(
        Output("sa_combo_phase3d", "figure", allow_duplicate=True),
        Output("sa_combo_series", "figure", allow_duplicate=True),
        Output("sa_combo_status", "children", allow_duplicate=True),
        Output("shared_traj", "data", allow_duplicate=True),
        Input("sa_combo_run", "n_clicks"),
        State("sa_combo_models", "value"),
        State("sa_combo_k", "value"),
        State("sa_combo_dt", "value"),
        State("sa_combo_T", "value"),
        State("sa_combo_jitter", "value"),
        State("sa_meta_store", "data"),
        prevent_initial_call=True,
    )
    def run_composite(
        n_clicks: int,
        models: List[str] | None,
        k: float,
        dt: float,
        T: float,
        jitter: float,
        meta: List[Dict[str, Any]] | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not models or len(models) < 2:
            return dash.no_update, dash.no_update, "Pick at least two attractors to couple.", dash.no_update
        meta_map = {m["key"]: m for m in (meta or [])}
        systems = []
        for key in models:
            info = meta_map.get(key)
            if not info:
                continue
            base_state = np.array(info.get("default_state", [1.0, 1.0, 1.0]), dtype=float)
            if jitter and jitter > 0:
                base_state = base_state + np.random.normal(0.0, jitter, size=base_state.shape)
            systems.append({"model": key, "params": {}, "x0": base_state.tolist()})
        if len(systems) < 2:
            return dash.no_update, dash.no_update, "Could not resolve selections.", dash.no_update
        payload = {"systems": systems, "coupling": float(k or 0.0), "dt": dt, "T": T}
        try:
            resp = requests.post(f"{API}/simulate/composite", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, dash.no_update, f"Composite run failed: {exc}", dash.no_update

        trajectories = [np.array(t, dtype=float) for t in data.get("trajectories", [])]
        time = np.array(data.get("time", []), dtype=float)
        labels = data.get("labels", ["X", "Y", "Z"])

        fig3 = go.Figure()
        colors = ["#2563eb", "#10b981", "#f97316", "#6366f1", "#ef4444", "#14b8a6", "#a855f7", "#0ea5e9"]
        for idx, traj in enumerate(trajectories):
            if traj.size == 0:
                continue
            fig3.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                    mode="lines",
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    name=f"{models[idx] if idx < len(models) else f'Run {idx+1}'}",
                )
            )
        fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_ts = go.Figure()
        for idx, traj in enumerate(trajectories):
            if traj.size == 0:
                continue
            for dim in range(min(3, traj.shape[1])):
                fig_ts.add_trace(
                    go.Scatter(
                        x=time,
                        y=traj[:, dim],
                        mode="lines",
                        name=f"{models[idx] if idx < len(models) else idx+1} · {labels[dim]}",
                        line=dict(width=1.6, color=colors[idx % len(colors)]),
                        opacity=max(0.4, 1.0 - 0.1 * dim),
                    )
                )
        fig_ts.update_layout(height=320, xaxis_title="t", legend=dict(orientation="h"))

        shared = trajectories[0].tolist() if trajectories else dash.no_update
        status = f"Coupled {len(trajectories)} systems; k={k}, dt={dt}, T={T}."
        return fig3, fig_ts, status, shared


register_page(
    "strange_attractors",
    name="Attractors",
    path="/attractors",
    layout=layout,
    register_callbacks=register_callbacks,
    order=12,
)
