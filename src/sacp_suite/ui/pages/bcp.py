from __future__ import annotations

import numpy as np
import requests
from dash import Input, Output, html, dcc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, get_shared_traj, PLOTLY_DARK_LAYOUT
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Bioelectric Section (BCP)", className="section-title"),
                    html.P("Section a trajectory or sample curve to inspect structure."),
                    html.Button("Run section", id="bcp_run", n_clicks=0, className="primary-btn"),
                    html.Div(id="bcp_shared_note", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="bcp_plot"), type="dot"),
                    html.Div(id="bcp_stats", style={"paddingTop": "6px"}),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("bcp_plot", "figure"),
        Output("bcp_stats", "children"),
        Output("bcp_shared_note", "children"),
        Input("bcp_run", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_bcp_section(n_clicks: int):
        if not n_clicks:
            raise PreventUpdate
        shared = get_shared_traj()
        if shared is not None and shared.shape[1] >= 2:
            pts = shared[:, [0, 2]] if shared.shape[1] > 2 else shared[:, :2]
            note = "Sectioned shared dataset."
        else:
            t = np.linspace(0, 2 * np.pi, 300)
            pts = np.stack([np.cos(t), np.sin(2 * t)], axis=1)
            note = "Sectioned sample curve."
        resp = requests.post(f"{API}/bcp/section", json={"points": pts.tolist()}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        sec = np.array(data.get("section", []), dtype=float)
        fig = go.Figure()
        if sec.size:
            fig.add_trace(go.Scatter(x=sec[:, 0], y=sec[:, 1], mode="markers", marker=dict(size=5)))
        fig.update_layout(**PLOTLY_DARK_LAYOUT)
        fig.update_layout(height=360, xaxis_title="X", yaxis_title="Z")
        return fig, f"Section points: {data.get('count', 0)}", note


register_page(
    "bcp",
    name="BCP",
    path="/bcp",
    layout=layout,
    register_callbacks=register_callbacks,
    order=80,
)
