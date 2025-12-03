from __future__ import annotations

import numpy as np
import requests
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, get_shared_traj
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("ABTC Lorenz Integrator", className="section-title"),
                    html.P("Integrate a Lorenz-like attractor via ABTC rk4 helper."),
                    html.Div(
                        [
                            html.Label("Steps"),
                            dcc.Input(id="abtc_steps", type="number", value=200, step=10),
                            html.Label("dt"),
                            dcc.Input(id="abtc_dt", type="number", value=0.01, step=0.001),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run ABTC", id="abtc_run", n_clicks=0, className="primary-btn"),
                    html.Div(id="abtc_shared_note", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="abtc_traj"), type="dot"),
                    dcc.Loading(dcc.Graph(id="abtc_series"), type="dot"),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("abtc_traj", "figure"),
        Output("abtc_series", "figure"),
        Output("abtc_shared_note", "children"),
        Input("abtc_run", "n_clicks"),
        State("abtc_steps", "value"),
        State("abtc_dt", "value"),
        prevent_initial_call=True,
    )
    def run_abtc(n_clicks: int, steps: int, dt: float):
        if not n_clicks:
            raise PreventUpdate
        shared = get_shared_traj()
        if shared is not None and getattr(shared, "size", 0):
            traj = np.array(shared, dtype=float)
            from_shared = True
        else:
            resp = requests.post(f"{API}/abtc/integrate", json={"steps": steps, "dt": dt}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            traj = np.array(data.get("trajectory", []), dtype=float)
            from_shared = False

        fig3 = go.Figure()
        if traj.size:
            fig3.add_trace(
                go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="lines", line=dict(width=2))
            )
        fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_ts = go.Figure()
        labels = ["X", "Y", "Z"]
        t = np.arange(traj.shape[0]) * dt
        for idx, label in enumerate(labels):
            fig_ts.add_trace(go.Scatter(x=t, y=traj[:, idx], name=label, mode="lines"))
        fig_ts.update_layout(height=300, xaxis_title="t")
        note = "Using shared dataset trajectory." if from_shared else "Using ABTC-generated trajectory."
        return fig3, fig_ts, note


register_page(
    "abtc",
    name="ABTC",
    path="/abtc",
    layout=layout,
    register_callbacks=register_callbacks,
    order=90,
)
