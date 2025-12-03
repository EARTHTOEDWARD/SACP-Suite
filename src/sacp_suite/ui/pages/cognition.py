from __future__ import annotations

import numpy as np
import requests
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from sacp_suite.ui.pages.common import (
    API_BASE,
    coerce_float,
    coerce_int,
    parse_numeric_list,
    parse_patterns_text,
)
from sacp_suite.ui.pages import register_page

API = API_BASE
DEFAULT_COG_ALPHABET = [-1.0, 0.0, 1.0]
DEFAULT_COG_PATTERNS = [
    [1.0, 0.0, -1.0, 0.0],
    [1.0, 1.0, 0.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
]


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Cognition · Memory", className="section-title"),
                    html.Div(
                        [
                            html.Label("Max lag"),
                            dcc.Input(id="cog-max-lag", type="number", value=100, step=10),
                            html.Label("Amplitude"),
                            dcc.Input(id="cog-amp", type="number", value=2.0, step=0.1),
                            html.Label("Base value"),
                            dcc.Input(id="cog-base-value", type="number", value=28.0, step=1.0),
                            html.Label("dt"),
                            dcc.Input(id="cog-dt", type="number", value=0.01, step=0.005),
                            html.Label("Trials"),
                            dcc.Input(id="cog-n-trials", type="number", value=4, step=1),
                            html.Label("Steps"),
                            dcc.Input(id="cog-n-steps", type="number", value=5000, step=100),
                            html.Label("Burn-in"),
                            dcc.Input(id="cog-burn-in", type="number", value=500, step=50),
                            html.Label("Alphabet (comma separated)"),
                            dcc.Input(id="cog-alphabet", type="text", value="-1,0,1", style={"width": "100%"}),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run memory profile", id="cog-run-memory", n_clicks=0, className="primary-btn"),
                    dcc.Loading(dcc.Graph(id="cog-memory-graph"), type="dot"),
                    dcc.Loading(dcc.Graph(id="cog-memory-sample"), type="dot"),
                    html.Div(id="cog-clc-summary", style={"paddingTop": "6px"}),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H3("Cognition · Discriminability", className="section-title"),
                    html.Div(
                        [
                            html.Label("Patterns (one per line, comma separated)"),
                            dcc.Textarea(
                                id="cog-patterns",
                                value="\n".join([", ".join(map(str, p)) for p in DEFAULT_COG_PATTERNS]),
                                style={"width": "100%", "minHeight": "120px"},
                            ),
                            html.Label("Trials per concept"),
                            dcc.Input(id="cog-disc-trials", type="number", value=100, step=10),
                            html.Label("Burn-in"),
                            dcc.Input(id="cog-disc-burn", type="number", value=500, step=50),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run discriminability", id="cog-run-disc", n_clicks=0, className="primary-btn"),
                    html.Div(id="cog-disc-summary", style={"paddingTop": "6px"}),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("cog-memory-graph", "figure"),
        Output("cog-memory-sample", "figure"),
        Output("cog-clc-summary", "children"),
        Input("cog-run-memory", "n_clicks"),
        State("cog-max-lag", "value"),
        State("cog-amp", "value"),
        State("cog-base-value", "value"),
        State("cog-dt", "value"),
        State("cog-n-trials", "value"),
        State("cog-n-steps", "value"),
        State("cog-burn-in", "value"),
        State("cog-alphabet", "value"),
        prevent_initial_call=True,
    )
    def run_cog_memory_experiment(
        n_clicks: int,
        max_lag: int | None,
        amp: float | None,
        base_value: float | None,
        dt_val: float | None,
        n_trials: int | None,
        n_steps: int | None,
        burn_in: int | None,
        alphabet_text: str | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        alphabet = parse_numeric_list(alphabet_text, DEFAULT_COG_ALPHABET)
        payload = {
            "max_lag": int(max_lag or 50),
            "amp": coerce_float(amp, 2.0),
            "base_value": coerce_float(base_value, 28.0),
            "dt": coerce_float(dt_val, 0.01),
            "n_trials": coerce_int(n_trials, 4, minimum=1),
            "n_steps": coerce_int(n_steps, 5000, minimum=200),
            "burn_in": coerce_int(burn_in, 500, minimum=0),
            "input_alphabet": alphabet,
        }
        resp = requests.post(f"{API}/cog/memory", json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        lags = data.get("lags", [])
        M_vals = data.get("M", [])
        mem_fig = {
            "data": [
                {
                    "x": lags,
                    "y": M_vals,
                    "mode": "lines+markers",
                    "name": "M(k)",
                }
            ],
            "layout": {
                "title": "Memory profile M(k)",
                "xaxis": {"title": "Lag k (steps)"},
                "yaxis": {"title": "Mutual information (bits)"},
            },
        }
        sample_u = data.get("sample_u", [])
        sample_y = data.get("sample_y", [])
        dt = float(data.get("dt", 1.0))
        n = min(len(sample_u), len(sample_y))
        times = (np.arange(n) * dt).tolist()
        sample_fig = {
            "data": [
                {
                    "x": times,
                    "y": sample_y[:n],
                    "mode": "lines",
                    "name": "Output (X)",
                },
                {
                    "x": times,
                    "y": sample_u[:n],
                    "mode": "lines",
                    "name": "Input u(t)",
                    "yaxis": "y2",
                },
            ],
            "layout": {
                "title": "Sample trial (first 1k points)",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "State"},
                "yaxis2": {"title": "Input", "overlaying": "y", "side": "right", "showgrid": False},
                "legend": {"orientation": "h"},
            },
        }
        clc_summary = (
            f"CLC · τ_past={data.get('tau_past', 0):.2f}s · "
            f"τ_future≈{data.get('tau_future', 0):.2f}s · "
            f"capture={data.get('capture', 0):.2f} · "
            f"radius≈{data.get('radius', 0):.2f} · "
            f"score={data.get('clc', 0):.2f}"
        )
        return mem_fig, sample_fig, clc_summary

    @app.callback(
        Output("cog-disc-summary", "children"),
        Input("cog-run-disc", "n_clicks"),
        State("cog-patterns", "value"),
        State("cog-disc-trials", "value"),
        State("cog-disc-burn", "value"),
        State("cog-amp", "value"),
        State("cog-base-value", "value"),
        State("cog-dt", "value"),
        prevent_initial_call=True,
    )
    def run_cog_discriminability(
        n_clicks: int,
        pattern_text: str | None,
        trials_per: int | None,
        burn_in: int | None,
        amp: float | None,
        base_value: float | None,
        dt_val: float | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        patterns = parse_patterns_text(pattern_text, DEFAULT_COG_PATTERNS)
        payload = {
            "patterns": [{"pattern": pattern} for pattern in patterns],
            "n_trials_per_concept": coerce_int(trials_per, 100, minimum=1),
            "burn_in": coerce_int(burn_in, 500, minimum=0),
            "amp": coerce_float(amp, 2.0),
            "base_value": coerce_float(base_value, 28.0),
            "dt": coerce_float(dt_val, 0.01),
        }
        resp = requests.post(f"{API}/cog/discriminability", json=payload, timeout=90)
        resp.raise_for_status()
        stats = resp.json()
        return html.Ul(
            [
                html.Li(f"K (concepts): {stats.get('K', 0):.0f}"),
                html.Li(f"T (length): {stats.get('T', 0):.0f}"),
                html.Li(f"Accuracy: {stats.get('accuracy', 0.0):.3f}"),
                html.Li(f"Pe: {stats.get('Pe', 0.0):.3f}"),
                html.Li(f"I_lower (bits): {stats.get('I_lower', 0.0):.3f}"),
                html.Li(f"D (normalized): {stats.get('D', 0.0):.3f}"),
            ],
            style={"lineHeight": "1.6"},
        )


register_page(
    "cognition",
    name="Cognition",
    path="/cog",
    layout=layout,
    register_callbacks=register_callbacks,
    order=50,
)
