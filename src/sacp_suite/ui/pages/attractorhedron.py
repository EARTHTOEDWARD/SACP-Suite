from __future__ import annotations

import numpy as np
import requests
from dash import Input, Output, html, dcc
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, get_shared_traj
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Attractorhedron — data-driven operator", className="section-title"),
                    html.P("Use the last Lorenz run or sent dataset to build an operator; inspect |lambda2|, gamma, and v2."),
                    html.Button("Build from last sim", id="build_op", n_clicks=0, className="primary-btn"),
                    html.Div(id="op_stats", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="v2_map"), type="dot"),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("op_stats", "children"),
        Output("v2_map", "figure"),
        Input("build_op", "n_clicks"),
        prevent_initial_call=True,
    )
    def build_operator(n_clicks: int):
        traj = get_shared_traj()
        if traj is None:
            return "Run a simulation first or send a dataset from the Datasets page.", go.Figure()

        traj = np.asarray(traj)
        XZ = traj[:, [0, 2]].tolist()
        build = requests.post(
            f"{API}/operator/build",
            json={"points": XZ, "nx": 25, "ny": 25},
            timeout=60,
        )
        build.raise_for_status()
        payload = build.json()
        P = payload["P"]

        ana = requests.post(
            f"{API}/operator/analyze",
            json={"P": P, "mean_return": 1.0, "nx": 25, "ny": 25},
            timeout=60,
        )
        ana.raise_for_status()
        stats = ana.json()
        v2 = np.array(stats["v2_field"], dtype=float)
        fig = go.Figure(data=go.Heatmap(z=v2, colorscale="Viridis"))
        fig.update_layout(height=420, xaxis_title="x-bin", yaxis_title="z-bin")
        summary = f"|lambda2| = {stats['abs_lambda2']:.3f}, gamma = {stats['gamma']:.3f}"
        return summary, fig


register_page(
    "attractorhedron",
    name="Attractorhedron",
    path="/attr",
    layout=layout,
    register_callbacks=register_callbacks,
    order=30,
)
