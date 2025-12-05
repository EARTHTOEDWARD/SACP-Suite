from __future__ import annotations

import json
from typing import Any, Dict

import requests
from dash import Input, Output, State, dcc, html, no_update

from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE, parse_numeric_list

API = f"{API_BASE}/api/v1/bouquet"


def _pretty(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def layout():
    return html.Div(
        [
            html.H3("Bouquet CLC"),
            html.P(
                "Run bouquet-stack simulations to test the cognitive speed bound and scan alpha for the knee."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Single run"),
                            html.Label("alpha"),
                            dcc.Input(id="bouquet-alpha", type="number", value=1.3, step=0.01),
                            html.Label("layers"),
                            dcc.Input(id="bouquet-n-layers", type="number", value=4, min=1, max=12),
                            html.Label("sites per layer"),
                            dcc.Input(id="bouquet-N", type="number", value=12, min=4, max=64),
                            html.Label("dt"),
                            dcc.Input(id="bouquet-dt", type="number", value=0.01, step=0.001),
                            html.Label("steps"),
                            dcc.Input(id="bouquet-steps", type="number", value=2000, step=100),
                            html.Label("log every"),
                            dcc.Input(id="bouquet-log-every", type="number", value=10, min=1),
                            html.Label("k_vert"),
                            dcc.Input(id="bouquet-kvert", type="number", value=0.05, step=0.01),
                            dcc.Checklist(
                                options=[{"label": "Enable self-tuning", "value": "yes"}],
                                value=[],
                                id="bouquet-self-tune",
                                style={"marginTop": "10px"},
                            ),
                            html.Label("target S_ring"),
                            dcc.Input(id="bouquet-target", type="number", value=1.5, step=0.1),
                            html.Label("control every (steps)"),
                            dcc.Input(id="bouquet-control-every", type="number", value=200, min=1),
                            html.Label("control window (logged frames)"),
                            dcc.Input(id="bouquet-control-window", type="number", value=40, min=2),
                            html.Button("Run bouquet", id="bouquet-run-btn", className="primary-btn", n_clicks=0),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.H4("Alpha scan"),
                            html.Label("alphas (comma separated)"),
                            dcc.Input(
                                id="bouquet-alpha-list",
                                type="text",
                                value="1.2,1.3,1.4,1.5,1.55,1.6",
                                style={"width": "100%"},
                            ),
                            html.Label("layers"),
                            dcc.Input(id="bouquet-scan-layers", type="number", value=4, min=1, max=12),
                            html.Label("sites per layer"),
                            dcc.Input(id="bouquet-scan-N", type="number", value=10, min=4, max=64),
                            html.Label("steps"),
                            dcc.Input(id="bouquet-scan-steps", type="number", value=1500, step=100),
                            html.Button("Scan alpha", id="bouquet-scan-btn", className="primary-btn", n_clicks=0),
                        ],
                        className="card",
                    ),
                ],
                className="mapping-grid",
            ),
            html.Div(id="bouquet-run-output", className="card"),
            html.Div(id="bouquet-scan-output", className="card"),
        ],
        id="bouquet-root",
    )


def register_callbacks(app):
    @app.callback(
        Output("bouquet-run-output", "children"),
        Input("bouquet-run-btn", "n_clicks"),
        State("bouquet-alpha", "value"),
        State("bouquet-n-layers", "value"),
        State("bouquet-N", "value"),
        State("bouquet-dt", "value"),
        State("bouquet-steps", "value"),
        State("bouquet-log-every", "value"),
        State("bouquet-kvert", "value"),
        State("bouquet-self-tune", "value"),
        State("bouquet-target", "value"),
        State("bouquet-control-every", "value"),
        State("bouquet-control-window", "value"),
        prevent_initial_call=True,
    )
    def _run(
        n_clicks,
        alpha,
        n_layers,
        N,
        dt,
        steps,
        log_every,
        kvert,
        self_tune_vals,
        target,
        control_every,
        control_window,
    ):
        if not n_clicks:
            return no_update
        payload: Dict[str, Any] = {
            "alpha": alpha,
            "n_layers": n_layers,
            "N": N,
            "dt": dt,
            "n_steps": steps,
            "log_every": log_every,
            "k_vert": kvert,
            "self_tune": bool(self_tune_vals),
            "target_S_ring": target,
            "control_every": control_every,
            "control_window": control_window,
        }
        try:
            resp = requests.post(f"{API}/run", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return html.Pre(_pretty(data))
        except Exception as exc:  # noqa: BLE001
            return html.Pre(f"Error: {exc}")

    @app.callback(
        Output("bouquet-scan-output", "children"),
        Input("bouquet-scan-btn", "n_clicks"),
        State("bouquet-alpha-list", "value"),
        State("bouquet-scan-layers", "value"),
        State("bouquet-scan-N", "value"),
        State("bouquet-scan-steps", "value"),
        prevent_initial_call=True,
    )
    def _scan(n_clicks, alpha_list, n_layers, N, steps):
        if not n_clicks:
            return no_update
        alphas = parse_numeric_list(alpha_list) if alpha_list else []
        payload = {
            "alphas": alphas,
            "n_layers": n_layers,
            "N": N,
            "n_steps": steps,
        }
        try:
            resp = requests.post(f"{API}/scan", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return html.Pre(_pretty(data))
        except Exception as exc:  # noqa: BLE001
            return html.Pre(f"Error: {exc}")


register_page(
    "bouquet",
    name="Bouquet CLC",
    path="/bouquet",
    layout=layout,
    register_callbacks=register_callbacks,
    order=165,
)
