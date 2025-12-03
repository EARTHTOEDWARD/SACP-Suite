from __future__ import annotations

import numpy as np
import requests
import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sacp_suite.modules.fractalhedron import build_fractalhedron_k, build_symbolic_sequence, fractal_face_flags
from sacp_suite.ui.pages.common import API_BASE, get_shared_traj, parse_label_list, parse_numeric_list
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("FractalHedron₂ — symbolic multifractal", className="section-title"),
                    html.P("Build symbolic multifractals from the latest trajectory or send cached operator results."),
                    html.Div(
                        [
                            html.Label("Coding spec"),
                            dcc.Dropdown(
                                id="fh-coding-spec",
                                options=[
                                    {"label": "Binary radius bins", "value": "radius_bins"},
                                    {"label": "X sign", "value": "x_sign"},
                                    {"label": "Z sign", "value": "z_sign"},
                                ],
                                value="x_sign",
                            ),
                            html.Label("Radius bins"),
                            dcc.Input(id="fh-radius-bins", placeholder="e.g. 0.2,0.5,1.0", style={"width": "100%"}),
                            html.Label("Radius labels"),
                            dcc.Input(id="fh-radius-labels", placeholder="Optional labels", style={"width": "100%"}),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Build FractalHedron", id="fh-run", n_clicks=0, className="primary-btn"),
                    html.Button("Use cached operator", id="fh-run-cached", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Div(id="fh-summary", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="fh-dq-graph"), type="dot"),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H3("Fractal LLM Reservoir", className="section-title"),
                    html.P("Integrate a Lorenz-like attractor through a small reservoir to produce symbolic outputs."),
                    html.Div(
                        [
                            html.Label("Reservoir size"),
                            dcc.Input(id="fllm_res_size", type="number", value=400, step=10),
                            html.Label("Coupling"),
                            dcc.Input(id="fllm_coupling", type="number", value=1.0, step=0.1),
                            html.Label("Prompt / text"),
                            dcc.Input(id="fllm_text", type="text", value="The strange attractor exhibits chaotic behavior", style={"width": "100%"}),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run Fractal LLM", id="fllm_run", n_clicks=0, className="primary-btn"),
                    html.Div(id="fllm_shared_note", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="fllm_traj"), type="dot"),
                    dcc.Loading(dcc.Graph(id="fllm_switch"), type="dot"),
                    html.Div(id="fllm_stats", style={"paddingTop": "6px"}),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("fh-summary", "children"),
        Output("fh-dq-graph", "figure"),
        Input("fh-run", "n_clicks"),
        Input("fh-run-cached", "n_clicks"),
        State("fh-coding-spec", "value"),
        State("fh-radius-bins", "value"),
        State("fh-radius-labels", "value"),
        prevent_initial_call=True,
    )
    def run_fractalhedron(
        run_clicks: int,
        cached_clicks: int,
        coding_spec: str,
        radius_bins: str | None,
        radius_labels: str | None,
    ):
        traj = get_shared_traj()
        if traj is None:
            return "Run a simulation or load a dataset first.", go.Figure()

        traj_arr = np.asarray(traj, dtype=float)
        if traj_arr.ndim != 2 or traj_arr.shape[1] < 3:
            return "Shared trajectory must have at least X,Y,Z columns.", go.Figure()

        coding_params = None
        if coding_spec == "radius_bins":
            params: dict = {}
            if radius_bins:
                bins_list = parse_numeric_list(radius_bins, [])
                if len(bins_list) < 2:
                    return "Provide at least two radius bin edges.", go.Figure()
                params["bins"] = bins_list
            if radius_labels:
                if "bins" not in params:
                    return "Specify radius bins before custom labels.", go.Figure()
                labels_list = parse_label_list(radius_labels, [])
                if len(labels_list) != len(params["bins"]) - 1:
                    return "Number of labels must equal len(bins) - 1.", go.Figure()
                params["labels"] = labels_list
            if params:
                coding_params = params

        try:
            if dash.callback_context.triggered_id == "fh-run-cached":
                payload = {
                    "section_hits": [],
                    "coding_spec": coding_spec,
                    "coding_params": coding_params or {},
                    "k": 2,
                    "Q": [0.0, 2.0],
                }
                resp = requests.post(f"{API}/fractalhedron/run", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                fh2 = {
                    "alphabet": data["alphabet"],
                    "D_q": {float(k): v for k, v in data["D_q"].items()},
                    "T_q": {float(k): v for k, v in data["T_q"].items()},
                    "ell_k": data["ell_k"],
                    "constraints": data["constraints"],
                }
                faces = data["faces"]
            else:
                sym_seq = build_symbolic_sequence(traj_arr, coding_spec=coding_spec, coding_params=coding_params)
                fh2 = build_fractalhedron_k(sym_seq, k=2, Q=(0.0, 2.0))
                faces = fractal_face_flags(fh2, eps_p=1e-5, eps_D=1e-2)
        except Exception as exc:  # noqa: BLE001
            return f"Could not build FractalHedron: {exc}", go.Figure()

        qs = sorted(fh2["D_q"].keys())
        fig = go.Figure()
        fig.add_bar(x=[str(q) for q in qs], y=[fh2["D_q"][q] for q in qs], name="D_q")
        fig.update_layout(
            height=320,
            title="Fractal spectrum",
            xaxis_title="q",
            yaxis_title="D_q",
            yaxis=dict(range=[0, 1.2]),
        )

        zero_words = faces["symbolic_zero_words"][:4]
        summary = html.Div(
            [
                html.Div(f"Alphabet: {', '.join(fh2['alphabet']) or 'n/a'}"),
                html.Div(f"ell_2: {fh2['ell_k']:.4f}"),
                html.Div(f"Monotone: {fh2['constraints']['monotone_ok']} · Bounds: {fh2['constraints']['bounds_ok']}"),
                html.Div(f"Zero-prob words (sample): {zero_words or '—'}"),
                html.Div(f"Monofractal pairs: {faces['monofractal_pairs'] or '—'}"),
                html.Div(
                    f"Near max dims: {faces['near_max_dim'] or '—'} · Near zero dims: {faces['near_zero_dim'] or '—'}"
                ),
            ],
            style={"fontFamily": "JetBrains Mono, monospace", "lineHeight": "1.6"},
        )
        return summary, fig

    @app.callback(
        Output("fllm_traj", "figure"),
        Output("fllm_switch", "figure"),
        Output("fllm_stats", "children"),
        Output("fllm_shared_note", "children"),
        Input("fllm_run", "n_clicks"),
        State("fllm_res_size", "value"),
        State("fllm_coupling", "value"),
        State("fllm_text", "value"),
        prevent_initial_call=True,
    )
    def run_fractalllm(n_clicks: int, res_size: int, coupling: float, text: str):
        shared = get_shared_traj()
        if shared is not None and getattr(shared, "size", 0):
            traj = np.array(shared, dtype=float)
            sw = np.zeros_like(traj)
            fig_main = go.Figure(
                data=[
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                        mode="lines",
                        line=dict(width=2),
                        name="Shared dataset",
                    )
                ]
            )
            fig_main.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
            fig_sw = go.Figure()
            note = "Using shared dataset trajectory."
            return fig_main, fig_sw, "Shared dataset visualized.", note

        payload = {"text": text or "", "reservoir_size": res_size, "coupling": coupling}
        resp = requests.post(f"{API}/fractalllm/process", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        traj = np.array(data.get("trajectory", []), dtype=float)
        sw = np.array(data.get("switch", []), dtype=float)

        fig_main = go.Figure()
        if traj.size:
            fig_main.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    line=dict(width=2),
                    name="Reservoir",
                )
            )
        fig_main.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_sw = go.Figure()
        if sw.size:
            fig_sw.add_trace(
                go.Scatter3d(
                    x=sw[:, 0],
                    y=sw[:, 1],
                    z=sw[:, 2],
                    mode="lines",
                    line=dict(width=2, color="firebrick"),
                    name="Switch",
                )
            )
        fig_sw.update_layout(height=320, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        note = ""
        return fig_main, fig_sw, f"Index={data.get('index', 0.0):.3f}, points={traj.shape[0]}", note


register_page(
    "fractal_llm",
    name="Fractal LLM",
    path="/fractalllm",
    layout=layout,
    register_callbacks=register_callbacks,
    order=40,
)
