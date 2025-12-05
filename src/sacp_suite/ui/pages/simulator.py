from __future__ import annotations

import numpy as np
import requests
import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import (
    API_BASE,
    PLOTLY_DARK_LAYOUT,
    get_shared_traj,
    parse_upload,
    set_shared_traj,
)
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Simulator — Lorenz 1963", className="section-title"),
                    html.Div(
                        [
                            html.Label("rho"),
                            dcc.Slider(id="rho", min=20, max=40, step=0.5, value=28),
                            html.Label("sigma"),
                            dcc.Slider(id="sigma", min=5, max=20, step=0.5, value=10),
                            html.Label("beta"),
                            dcc.Slider(id="beta", min=1.5, max=3.0, step=0.05, value=8 / 3),
                            html.Label("Time horizon (T)"),
                            dcc.Slider(id="T", min=20, max=120, step=5, value=60),
                            html.Label("dt"),
                            dcc.Slider(
                                id="dt",
                                min=0.001,
                                max=0.05,
                                step=0.001,
                                value=0.01,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"padding": "6px"},
                    ),
                    html.Button("Run", id="run", n_clicks=0, className="primary-btn"),
                    html.Button("Show shared trajectory", id="show_shared_traj", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Div(
                        [
                            dcc.Loading(dcc.Graph(id="phase3d"), type="dot"),
                            dcc.Loading(dcc.Graph(id="series"), type="dot"),
                            html.Div(id="lle_out", style={"paddingTop": "6px", "fontWeight": "600"}),
                        ],
                        className="section-grid",
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("Upload your data to compute LLE", className="section-title"),
                    html.P("Upload a CSV with one numeric column (or pick the first column).", style={"color": "#4b5563"}),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and drop or ", html.B("select a CSV")]),
                        className="upload-box",
                        multiple=False,
                    ),
                    html.Button("Compute LLE from file", id="lle_from_file_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="lle_file_out", style={"paddingTop": "6px", "fontWeight": "600"}),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("phase3d", "figure", allow_duplicate=True),
        Output("series", "figure", allow_duplicate=True),
        Output("lle_out", "children", allow_duplicate=True),
        Input("run", "n_clicks"),
        State("rho", "value"),
        State("sigma", "value"),
        State("beta", "value"),
        State("T", "value"),
        State("dt", "value"),
        prevent_initial_call=True,
    )
    def do_run(n_clicks: int, rho: float, sigma: float, beta: float, T: float, dt: float):
        req = {
            "model": "lorenz63",
            "params": {"rho": rho, "sigma": sigma, "beta": beta},
            "T": T,
            "dt": dt,
        }
        resp = requests.post(f"{API}/simulate", json=req, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        traj = np.array(data["trajectory"], dtype=float)
        time = np.array(data["time"], dtype=float)

        lle_resp = requests.post(
            f"{API}/metrics/lle",
            json={"series": traj[:, 0].tolist()},
            timeout=60,
        )
        lle_resp.raise_for_status()
        lle_val = float(lle_resp.json()["lle"])

        fig3 = go.Figure(
            data=[
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    line=dict(width=2),
                )
            ]
        )
        fig3.update_layout(**PLOTLY_DARK_LAYOUT)
        fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_ts = go.Figure()
        labels = data.get("labels", ["X", "Y", "Z"])
        for idx, label in enumerate(labels):
            fig_ts.add_trace(go.Scatter(x=time, y=traj[:, idx], name=label, mode="lines"))
        fig_ts.update_layout(**PLOTLY_DARK_LAYOUT)
        fig_ts.update_layout(height=300, xaxis_title="t")

        set_shared_traj(traj)
        return fig3, fig_ts, f"Estimated LLE (Rosenstein): {lle_val:.3f}"

    @app.callback(
        Output("phase3d", "figure", allow_duplicate=True),
        Output("series", "figure", allow_duplicate=True),
        Output("lle_out", "children", allow_duplicate=True),
        Input("show_shared_traj", "n_clicks"),
        prevent_initial_call=True,
    )
    def show_shared(n_clicks: int):
        traj = get_shared_traj()
        if traj is None or getattr(traj, "size", 0) == 0:
            return dash.no_update, dash.no_update, "No shared dataset available."
        traj = np.array(traj, dtype=float)
        time = np.arange(traj.shape[0])
        lle_resp = requests.post(
            f"{API}/metrics/lle",
            json={"series": traj[:, 0].tolist()},
            timeout=60,
        )
        lle_resp.raise_for_status()
        lle_val = float(lle_resp.json()["lle"])

        fig3 = go.Figure(
            data=[
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                    mode="lines",
                    line=dict(width=2),
                )
            ]
        )
        fig3.update_layout(**PLOTLY_DARK_LAYOUT)
        fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_ts = go.Figure()
        labels = ["X", "Y", "Z"]
        for idx, label in enumerate(labels):
            if idx < traj.shape[1]:
                fig_ts.add_trace(go.Scatter(x=time, y=traj[:, idx], name=label, mode="lines"))
        fig_ts.update_layout(**PLOTLY_DARK_LAYOUT)
        fig_ts.update_layout(height=300, xaxis_title="t")
        return fig3, fig_ts, f"Shared dataset LLE (X): {lle_val:.3f}"

    @app.callback(
        Output("lle_file_out", "children"),
        Input("lle_from_file_btn", "n_clicks"),
        State("upload-data", "contents"),
        prevent_initial_call=True,
    )
    def compute_lle_from_file(n_clicks: int, contents: str | None):
        if not contents:
            return "Upload a CSV first."
        try:
            series = parse_upload(contents)
        except Exception as exc:  # noqa: BLE001
            return f"Could not read file: {exc}"
        resp = requests.post(
            f"{API}/metrics/lle",
            json={"series": series.tolist()},
            timeout=60,
        )
        resp.raise_for_status()
        val = float(resp.json()["lle"])
        return f"Uploaded series LLE: {val:.3f}"


register_page(
    "simulator",
    name="Simulator",
    path="/sim",
    layout=layout,
    register_callbacks=register_callbacks,
    order=10,
)
