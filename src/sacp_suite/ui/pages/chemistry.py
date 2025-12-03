from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import requests
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

from sacp_suite.ui.pages import register_page

_API_BASE = os.getenv("SACP_API_BASE_URL", "http://127.0.0.1:8000")
_CHEM_SIM_URL = f"{_API_BASE}/api/v1/chemistry/simulate"


def layout():
    return html.Div(
        [
            html.H2("Chemistry / ADR Bioelectric"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("t_max"),
                            dcc.Input(
                                id="chemistry-tmax",
                                type="number",
                                value=100.0,
                                step=1.0,
                            ),
                        ],
                        className="chemistry-control",
                    ),
                    html.Div(
                        [
                            html.Label("dt"),
                            dcc.Input(
                                id="chemistry-dt",
                                type="number",
                                value=0.01,
                                step=0.005,
                            ),
                        ],
                        className="chemistry-control",
                    ),
                    html.Div(
                        [
                            html.Label("Initial state (comma-separated floats)"),
                            dcc.Input(
                                id="chemistry-initial",
                                type="text",
                                value="0.0, 0.1, -0.2",
                                style={"width": "100%"},
                            ),
                        ],
                        className="chemistry-control",
                    ),
                    html.Button(
                        "Run chemistry simulation",
                        id="chemistry-run",
                        n_clicks=0,
                        className="chemistry-run-button",
                    ),
                ],
                className="chemistry-controls",
            ),
            html.Div(id="chemistry-error", className="chemistry-error"),
            dcc.Loading(
                id="chemistry-loading",
                type="default",
                children=dcc.Graph(
                    id="chemistry-trajectory",
                    figure=go.Figure(layout={"title": "ADR trajectory"}),
                    style={"height": "480px"},
                ),
            ),
        ],
        id="chemistry-root",
        className="chemistry-page",
    )


def _parse_initial_state(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        s = chunk.strip()
        if not s:
            continue
        values.append(float(s))
    if not values:
        raise ValueError("No values provided")
    return values


def _call_chemistry_api(payload: Dict[str, Any]) -> Tuple[List[float], List[List[float]], Dict[str, Any]]:
    resp = requests.post(_CHEM_SIM_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    t = data.get("t", [])
    x = data.get("x", []) or data.get("states", [])
    meta = {
        "lle": data.get("lle") or data.get("lyapunov"),
        **(data.get("metadata") or {}),
    }
    return t, x, meta


def register_callbacks(app):
    @app.callback(
        Output("chemistry-trajectory", "figure"),
        Output("chemistry-error", "children"),
        Input("chemistry-run", "n_clicks"),
        State("chemistry-tmax", "value"),
        State("chemistry-dt", "value"),
        State("chemistry-initial", "value"),
        prevent_initial_call=True,
    )
    def _run_chemistry(n_clicks, t_max, dt, initial_raw):
        if not n_clicks:
            raise PreventUpdate

        if t_max is None or dt is None or not initial_raw:
            return no_update, "Please fill in t_max, dt and initial state."

        try:
            initial_state = _parse_initial_state(initial_raw)
        except Exception as exc:  # noqa: BLE001
            return no_update, f"Could not parse initial state: {exc}"

        payload = {
            "t_max": t_max,
            "dt": dt,
            "initial_state": initial_state,
        }

        try:
            t, x, meta = _call_chemistry_api(payload)
        except Exception as exc:  # noqa: BLE001
            return no_update, f"Chemistry simulation failed: {exc}"

        if not t or not x:
            return no_update, "Chemistry simulation returned no data."

        fig = go.Figure()
        n_state = len(x[0])
        series = list(zip(*x))

        for idx, series_data in enumerate(series[: min(n_state, 3)]):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=series_data,
                    mode="lines",
                    name=f"state[{idx}]",
                )
            )

        title = "ADR trajectory"
        if meta.get("lle") is not None:
            title += f" (LLE={meta['lle']:.3g})"

        fig.update_layout(
            title=title,
            xaxis_title="t",
            yaxis_title="state",
        )

        return fig, ""


register_page(
    "chemistry",
    name="Chemistry",
    path="/chemistry",
    layout=layout,
    register_callbacks=register_callbacks,
    order=110,
)
