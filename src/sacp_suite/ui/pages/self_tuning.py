from __future__ import annotations

import numpy as np
import requests
import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, PLOTLY_DARK_LAYOUT
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Self-tuning Lorenz", className="section-title"),
                    html.P("Run live self-tuning sessions or a static demo."),
                    html.Div(
                        [
                            html.Label("Steps"),
                            dcc.Input(id="self-tune-steps", type="number", value=2000, step=100),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Start live session", id="self-tune-run", n_clicks=0, className="primary-btn"),
                    html.Div(id="self-tune-session-id", style={"paddingTop": "6px"}),
                    dcc.Interval(id="self-tune-timer", interval=2000, n_intervals=0, disabled=True),
                    html.Div(id="self-tune-summary", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="self-tune-lambda"), type="dot"),
                    dcc.Loading(dcc.Graph(id="self-tune-sr"), type="dot"),
                    dcc.Loading(dcc.Graph(id="self-tune-attractor"), type="dot"),
                    html.Hr(),
                    html.Button("Run demo (batch)", id="self-tune-demo", n_clicks=0, className="primary-btn"),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("self-tune-session-id", "children"),
        Output("self-tune-timer", "disabled", allow_duplicate=True),
        Output("self-tune-summary", "children", allow_duplicate=True),
        Input("self-tune-run", "n_clicks"),
        State("self-tune-steps", "value"),
        prevent_initial_call=True,
    )
    def start_self_tune_session(n_clicks: int, steps: int):
        payload = {"dt": 0.01}
        resp = requests.post(f"{API}/self-tuning/sessions", json=payload, timeout=30)
        resp.raise_for_status()
        session_id = resp.json()["session_id"]
        return session_id, False, f"Session started ({session_id[:8]})"

    @app.callback(
        Output("self-tune-lambda", "figure", allow_duplicate=True),
        Output("self-tune-sr", "figure", allow_duplicate=True),
        Output("self-tune-attractor", "figure", allow_duplicate=True),
        Output("self-tune-summary", "children", allow_duplicate=True),
        Output("self-tune-timer", "disabled", allow_duplicate=True),
        Input("self-tune-timer", "n_intervals"),
        State("self-tune-session-id", "children"),
        State("self-tune-steps", "value"),
        prevent_initial_call=True,
    )
    def update_self_tune_live(n_intervals: int, session_id: str, steps: int):
        if not session_id:
            raise PreventUpdate
        resp = requests.post(
            f"{API}/self-tuning/sessions/step",
            json={"session_id": session_id, "steps": max(steps // 5, 50)},
            timeout=60,
        )
        if resp.status_code != 200:
            return dash.no_update, dash.no_update, dash.no_update, f"Session error: {resp.text}", True
        data = resp.json()
        lambdas = data.get("lambdas", [])
        sr_hist = data.get("spectral_radius", [])
        fig_lambda = go.Figure()
        if lambdas:
            fig_lambda.add_trace(go.Scatter(x=list(range(len(lambdas))), y=lambdas, mode="lines", name="λmax"))
        fig_lambda.update_layout(height=320, title=f"λ₁(t) — step {data['step']}", xaxis_title="step", yaxis_title="λ₁")

        fig_sr = go.Figure()
        if sr_hist:
            fig_sr.add_trace(go.Scatter(x=list(range(len(sr_hist))), y=sr_hist, mode="lines", name="spectral radius"))
        fig_sr.update_layout(height=320, title="Spectral radius", xaxis_title="step", yaxis_title="ρ scale")

        fig_attractor = go.Figure()
        state = data.get("state")
        if state:
            fig_attractor.add_trace(go.Scatter3d(x=[state[0]], y=[state[1]], z=[state[2]], mode="markers"))
            fig_attractor.update_layout(height=360, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        summary = (
            f"step={data['step']} · λ₁={lambdas[-1]:.4f} · regime={data.get('regime') or 'n/a'} · ρ scale={sr_hist[-1]:.3f}"
            if lambdas
            else f"step={data['step']}"
        )
        disable_timer = data["step"] >= (steps or 2000)
        return fig_lambda, fig_sr, fig_attractor, summary, disable_timer

    @app.callback(
        Output("self-tune-lambda", "figure", allow_duplicate=True),
        Output("self-tune-sr", "figure", allow_duplicate=True),
        Output("self-tune-attractor", "figure", allow_duplicate=True),
        Output("self-tune-summary", "children", allow_duplicate=True),
        Input("self-tune-demo", "n_clicks"),
        State("self-tune-steps", "value"),
        prevent_initial_call=True,
    )
    def run_self_tuning_demo(n_clicks: int, steps: int):
        payload = {"num_steps": steps or 2000, "record_states": True}
        resp = requests.post(f"{API}/self-tuning/lorenz", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        lambdas = data.get("lambdas", [])
        regimes = data.get("regimes", [])
        sr_hist = data.get("spectral_radius", [])
        states = np.array(data.get("states", []), dtype=float)

        x_series = list(range(len(lambdas)))
        fig_lambda = go.Figure()
        if lambdas:
            fig_lambda.add_trace(go.Scatter(x=x_series, y=lambdas, mode="lines", name="λmax"))
        fig_lambda.update_layout(height=320, title="λ₁(t)", xaxis_title="step", yaxis_title="λ₁")

        fig_sr = go.Figure()
        if sr_hist:
            fig_sr.add_trace(go.Scatter(x=list(range(len(sr_hist))), y=sr_hist, mode="lines", name="spectral radius"))
        fig_sr.update_layout(height=320, title="Spectral radius", xaxis_title="step", yaxis_title="ρ scale")

        fig_attractor = go.Figure()
        if states.size:
            sample = states[:: max(len(states) // 500, 1)]
            fig_attractor.add_trace(
                go.Scatter3d(
                    x=sample[:, 0],
                    y=sample[:, 1],
                    z=sample[:, 2],
                    mode="lines",
                    line=dict(width=2),
                )
            )
            fig_attractor.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        if lambdas:
            last_lambda = lambdas[-1]
            last_regime = regimes[-1] if regimes else None
            final_sr = sr_hist[-1] if sr_hist else float("nan")
            summary = f"λ₁ final = {last_lambda:.4f} · regime = {last_regime or 'n/a'} · ρ scale = {final_sr:.3f}"
        else:
            summary = "Run the demo to see diagnostics."
        return fig_lambda, fig_sr, fig_attractor, summary


register_page(
    "self_tuning",
    name="Self Tuning",
    path="/self-tune",
    layout=layout,
    register_callbacks=register_callbacks,
    order=70,
)
