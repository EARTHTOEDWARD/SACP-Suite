"""Dash multi-page UI for SACP Suite."""

from __future__ import annotations

import os
from typing import Tuple

import dash
import numpy as np
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html

API = f"http://{os.getenv('SACP_BIND', '127.0.0.1')}:{os.getenv('SACP_PORT', '8000')}"

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="SACP Suite")
server = app.server

NAV = html.Nav(
    [
        dcc.Link("Home", href="/"),
        html.Span(" | "),
        dcc.Link("Simulator", href="/sim"),
        html.Span(" | "),
        dcc.Link("Attractorhedron", href="/attr"),
    ],
    style={"padding": "8px", "fontFamily": "system-ui"},
)

app.layout = html.Div(
    [dcc.Location(id="url"), NAV, html.Div(id="page")],
    style={"maxWidth": "1200px", "margin": "0 auto"},
)


def page_home():
    return html.Div(
        [
            html.H2("SACP Suite Starter"),
            html.P("Unified API + UI with clean, per-module pages."),
            html.Ul(
                [
                    html.Li("Simulator: Run Lorenz and compute lambda_1 (LLE)"),
                    html.Li("Attractorhedron: Build Ulam operator and inspect spectrum"),
                ]
            ),
            html.Hr(),
            html.Pre(f"API -> {API}", style={"background": "#f6f8fa", "padding": "8px"}),
        ]
    )


def page_sim():
    return html.Div(
        [
            html.H3("Simulator — Lorenz 1963"),
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
            html.Button("Run", id="run", n_clicks=0),
            dcc.Loading(dcc.Graph(id="phase3d"), type="dot"),
            dcc.Loading(dcc.Graph(id="series"), type="dot"),
            html.Div(id="lle_out", style={"paddingTop": "6px", "fontWeight": "600"}),
        ]
    )


def page_attr():
    return html.Div(
        [
            html.H3("Attractorhedron — data-driven operator"),
            html.P("Use the last Lorenz run (X,Z) to build an operator; inspect |lambda2|, gamma, and v2."),
            html.Button("Build from last sim", id="build_op", n_clicks=0),
            html.Div(id="op_stats", style={"paddingTop": "6px"}),
            dcc.Loading(dcc.Graph(id="v2_map"), type="dot"),
        ]
    )


@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(pathname: str):
    if pathname == "/sim":
        return page_sim()
    if pathname == "/attr":
        return page_attr()
    return page_home()


@app.callback(
    Output("phase3d", "figure"),
    Output("series", "figure"),
    Output("lle_out", "children"),
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
    fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig_ts = go.Figure()
    labels = data.get("labels", ["X", "Y", "Z"])
    for idx, label in enumerate(labels):
        fig_ts.add_trace(go.Scatter(x=time, y=traj[:, idx], name=label, mode="lines"))
    fig_ts.update_layout(height=300, xaxis_title="t")

    server.last_traj = traj
    return fig3, fig_ts, f"Estimated LLE (Rosenstein): {lle_val:.3f}"


@app.callback(
    Output("op_stats", "children"),
    Output("v2_map", "figure"),
    Input("build_op", "n_clicks"),
    prevent_initial_call=True,
)
def build_operator(n_clicks: int):
    traj = getattr(server, "last_traj", None)
    if traj is None:
        return "Run a simulation first.", go.Figure()

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


def main() -> None:
    host = os.getenv("SACP_UI_HOST", "127.0.0.1")
    port = int(os.getenv("SACP_UI_PORT", "8050"))
    app.run_server(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
