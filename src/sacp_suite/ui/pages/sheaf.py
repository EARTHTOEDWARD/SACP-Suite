from __future__ import annotations

import numpy as np
import requests
from dash import Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate
import dash
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, parse_numeric_list
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Lorenz Sheaf Sweep", className="section-title"),
                    html.P("Sweep ρ and show which attractor class persists; gluing failures show up as obstructions."),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("ρ values (comma separated)"),
                                    dcc.Input(
                                        id="sheaf-rhos",
                                        type="text",
                                        value="0.5,1,5,10,15,20,24,28,35",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 2},
                            ),
                            html.Div(
                                [
                                    html.Label("Steps"),
                                    dcc.Input(id="sheaf-steps", type="number", value=6000, min=500, step=500),
                                    html.Label("Burn-in"),
                                    dcc.Input(id="sheaf-burn", type="number", value=1000, min=0, step=100),
                                    html.Label("dt"),
                                    dcc.Input(id="sheaf-dt", type="number", value=0.01, min=0.001, step=0.001),
                                ],
                                style={
                                    "flex": 1,
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(2, minmax(0,1fr))",
                                    "gap": "6px",
                                },
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "padding": "6px"},
                    ),
                    html.Button("Run sheaf sweep", id="sheaf-run", n_clicks=0, className="primary-btn"),
                    html.Div(id="sheaf-summary", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="sheaf-lambda-fig"), type="dot"),
                    dcc.Loading(dcc.Graph(id="sheaf-clc-fig"), type="dot"),
                    html.Div(
                        dash_table.DataTable(
                            id="sheaf-sections-table",
                            columns=[
                                {"name": "start", "id": "start"},
                                {"name": "end", "id": "end"},
                                {"name": "class", "id": "class_label"},
                            ],
                            page_size=10,
                            style_table={"maxHeight": "240px", "overflowY": "auto"},
                        ),
                        style={"paddingTop": "6px"},
                    ),
                    html.Div(id="sheaf-obstructions", style={"paddingTop": "6px"}),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("sheaf-lambda-fig", "figure"),
        Output("sheaf-clc-fig", "figure"),
        Output("sheaf-sections-table", "data"),
        Output("sheaf-summary", "children"),
        Output("sheaf-obstructions", "children"),
        Input("sheaf-run", "n_clicks"),
        State("sheaf-rhos", "value"),
        State("sheaf-steps", "value"),
        State("sheaf-burn", "value"),
        State("sheaf-dt", "value"),
        prevent_initial_call=True,
    )
    def run_sheaf(n_clicks: int, rho_text: str, steps: int, burn: int, dt: float):
        if not n_clicks:
            raise PreventUpdate
        rhos = parse_numeric_list(rho_text, [])
        if not rhos:
            return dash.no_update, dash.no_update, dash.no_update, "Provide at least one ρ value.", ""
        payload = {
            "rhos": rhos,
            "steps": int(steps or 6000),
            "burn_in": int(burn or 1000),
            "dt": float(dt or 0.01),
        }
        resp = requests.post(f"{API}/sheaf/lorenz", json=payload, timeout=120)
        if resp.status_code != 200:
            return dash.no_update, dash.no_update, dash.no_update, f"Error: {resp.text}", ""
        data = resp.json()
        samples = data.get("samples", [])
        sections = data.get("sections", [])
        obstructions = data.get("obstructions", [])

        rho_vals = [s["rho"] for s in samples]
        lambdas = [s["lambda_max"] for s in samples]
        classes = [s["class_label"] for s in samples]
        clc_vals = [s.get("clc", 0) for s in samples]

        fig_lambda = go.Figure()
        fig_lambda.add_trace(
            go.Scatter(
                x=rho_vals,
                y=lambdas,
                mode="markers+lines",
                marker=dict(color="firebrick"),
                text=classes,
                name="λ₁/dt",
            )
        )
        fig_lambda.update_layout(title="λ₁ vs ρ", xaxis_title="ρ", yaxis_title="λ₁ (per unit time)", height=320)

        fig_clc = go.Figure()
        fig_clc.add_trace(
            go.Bar(
                x=rho_vals,
                y=clc_vals,
                name="CLC score",
                marker_color="seagreen",
            )
        )
        fig_clc.update_layout(title="CLC vs ρ", xaxis_title="ρ", yaxis_title="CLC", height=320)

        summary = f"{len(sections)} sections; {len(obstructions)} obstructions (class changes)." if samples else "No samples"
        obs_text = ", ".join(f"[{o['left']}, {o['right']}]: {o['reason']}" for o in obstructions)
        return fig_lambda, fig_clc, sections, summary, obs_text


register_page(
    "sheaf",
    name="Sheaf",
    path="/sheaf",
    layout=layout,
    register_callbacks=register_callbacks,
    order=20,
)
