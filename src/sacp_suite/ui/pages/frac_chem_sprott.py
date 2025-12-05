from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE, set_shared_traj, PLOTLY_DARK_LAYOUT

API = API_BASE


def layout():
    return html.Div(
        [
            html.H3("Fractional Chemical Sprott (Hidden Attractor)"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Parameters", className="section-title"),
                            dcc.Slider(
                                id="fcs-q-slider",
                                min=0.55,
                                max=1.0,
                                step=0.01,
                                value=0.95,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div("q (fractional order)", className="param-label"),
                            dcc.Slider(
                                id="fcs-k1-slider",
                                min=0.0,
                                max=0.07,
                                step=0.001,
                                value=0.006,
                            ),
                            html.Div("k1 (source X)", className="param-label"),
                            dcc.Input(
                                id="fcs-m-input",
                                type="number",
                                value=0.0,
                                debounce=True,
                                placeholder="offset m (z -> z + m)",
                                className="param-input",
                            ),
                            html.Br(),
                            html.Div("Initial conditions (X0, Y0, Z0)"),
                            dcc.Input(id="fcs-x0", type="number", value=0.1, step=0.01),
                            dcc.Input(id="fcs-y0", type="number", value=0.1, step=0.01),
                            dcc.Input(id="fcs-z0", type="number", value=0.1, step=0.01),
                            html.Br(),
                            html.Button(
                                "Simulate",
                                id="fcs-run-btn",
                                className="primary-btn",
                                n_clicks=0,
                            ),
                            html.Button(
                                "Scan q / k1",
                                id="fcs-scan-btn",
                                className="primary-btn",
                                n_clicks=0,
                                style={"marginLeft": "0.5rem"},
                            ),
                            html.Div(id="fcs-status", className="status-text"),
                        ],
                        className="card",
                        style={"flex": "0 0 340px"},
                    ),
                    html.Div(
                        [
                            dcc.Tabs(
                                id="fcs-tabs",
                                value="traj",
                                children=[
                                    dcc.Tab(label="Trajectory", value="traj"),
                                    dcc.Tab(label="Bifurcation", value="bif"),
                                    dcc.Tab(label="Complexity", value="complexity"),
                                    dcc.Tab(label="Currents", value="currents"),
                                ],
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="fcs-traj-timeseries"),
                                    dcc.Graph(id="fcs-traj-phase3d"),
                                ],
                                id="fcs-tab-traj",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="fcs-bif-q"),
                                    dcc.Graph(id="fcs-bif-k1"),
                                ],
                                id="fcs-tab-bif",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="fcs-complexity-heatmap-se"),
                                    dcc.Graph(id="fcs-complexity-heatmap-c0"),
                                ],
                                id="fcs-tab-complexity",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="fcs-currents-x"),
                                    dcc.Graph(id="fcs-currents-yz"),
                                ],
                                id="fcs-tab-currents",
                                style={"display": "none"},
                            ),
                        ],
                        style={"flex": "1 1 auto"},
                    ),
                ],
                style={"display": "flex", "gap": "1rem"},
            ),
        ],
        id="frac-chem-sprott-root",
    )


def register_callbacks(app):
    @app.callback(
        Output("fcs-tab-traj", "style"),
        Output("fcs-tab-bif", "style"),
        Output("fcs-tab-complexity", "style"),
        Output("fcs-tab-currents", "style"),
        Input("fcs-tabs", "value"),
    )
    def _switch_tab(tab):
        hidden = {"display": "none"}
        if tab == "traj":
            return {}, hidden, hidden, hidden
        if tab == "bif":
            return hidden, {}, hidden, hidden
        if tab == "complexity":
            return hidden, hidden, {}, hidden
        if tab == "currents":
            return hidden, hidden, hidden, {}
        return {}, hidden, hidden, hidden

    @app.callback(
        Output("fcs-traj-timeseries", "figure"),
        Output("fcs-traj-phase3d", "figure"),
        Output("fcs-currents-x", "figure"),
        Output("fcs-currents-yz", "figure"),
        Output("fcs-status", "children"),
        Input("fcs-run-btn", "n_clicks"),
        State("fcs-q-slider", "value"),
        State("fcs-k1-slider", "value"),
        State("fcs-m-input", "value"),
        State("fcs-x0", "value"),
        State("fcs-y0", "value"),
        State("fcs-z0", "value"),
        prevent_initial_call=True,
    )
    def _run_sim(n_clicks, q, k1, m, x0, y0, z0):
        if not n_clicks:
            raise PreventUpdate

        payload = {
            "params": {"q": q, "k1": k1, "m": m},
            "x0": [x0, y0, z0],
            "t_max": 200.0,
            "dt": 0.002,
            "t_transient": 40.0,
            "sample_stride": 5,
            "return_currents": True,
            "return_0_1_test": False,
            "return_complexity": False,
            "observable": "x",
        }

        try:
            resp = requests.post(f"{API}/api/v1/frac-chem-sprott/simulate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            empty = go.Figure()
            return empty, empty, empty, empty, f"Simulation failed: {exc}"

        t = data["traj"]["t"]
        X = data["traj"]["x"]
        Y = data["traj"]["y"]
        Z = data["traj"]["z"]
        set_shared_traj({"t": t, "x": X, "y": Y, "z": Z, "meta": data.get("params", {})})

        fig_ts = go.Figure()
        fig_ts.add_scatter(x=t, y=X, mode="lines", name="X")
        fig_ts.add_scatter(x=t, y=Y, mode="lines", name="Y")
        fig_ts.add_scatter(x=t, y=Z, mode="lines", name="Z")
        fig_ts.update_layout(title="X, Y, Z vs time", xaxis_title="t", yaxis_title="concentration")

        fig_3d = go.Figure(
            data=[
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="lines",
                )
            ]
        )
        fig_3d.update_layout(
            title="Phase portrait (X, Y, Z)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        curr = data.get("currents")
        if curr:
            JS = np.asarray(curr.get("JS", []), dtype=float)
            JC = np.asarray(curr.get("JC", []), dtype=float)
            JR = np.asarray(curr.get("JR", []), dtype=float)
            JE = np.asarray(curr.get("JE", []), dtype=float)

            fig_cx = go.Figure()
            if JS.size:
                fig_cx.add_scatter(x=t[: JS.shape[0]], y=JS[:, 0], mode="lines", name="JS_X")
            if JC.size:
                fig_cx.add_scatter(x=t[: JC.shape[0]], y=JC[:, 0], mode="lines", name="JC_X")
            if JR.size:
                fig_cx.add_scatter(x=t[: JR.shape[0]], y=JR[:, 0], mode="lines", name="JR_X")
            if JE.size:
                fig_cx.add_scatter(x=t[: JE.shape[0]], y=JE[:, 0], mode="lines", name="JE_X")
            fig_cx.update_layout(title="Currents for X", xaxis_title="t", yaxis_title="rate")

            fig_cyz = go.Figure()
            if JS.size:
                fig_cyz.add_scatter(x=t[: JS.shape[0]], y=JS[:, 1], mode="lines", name="JS_Y")
            if JC.size:
                fig_cyz.add_scatter(x=t[: JC.shape[0]], y=JC[:, 1], mode="lines", name="JC_Y")
            if JR.size:
                fig_cyz.add_scatter(x=t[: JR.shape[0]], y=JR[:, 1], mode="lines", name="JR_Y")
            if JE.size:
                fig_cyz.add_scatter(x=t[: JE.shape[0]], y=JE[:, 1], mode="lines", name="JE_Y")
            fig_cyz.update_layout(title="Currents for Y", xaxis_title="t", yaxis_title="rate")
        else:
            fig_cx = go.Figure()
            fig_cyz = go.Figure()

        lle_val = data.get("lle")
        status = f"Sim OK: lle={lle_val:.4f}" if isinstance(lle_val, (int, float)) else "Sim OK"
        if data.get("diverged"):
            status = "Simulation diverged early; showing partial trajectory."
        return fig_ts, fig_3d, fig_cx, fig_cyz, status

    @app.callback(
        Output("fcs-bif-q", "figure"),
        Output("fcs-bif-k1", "figure"),
        Output("fcs-complexity-heatmap-se", "figure"),
        Output("fcs-complexity-heatmap-c0", "figure"),
        Input("fcs-scan-btn", "n_clicks"),
        State("fcs-k1-slider", "value"),
        State("fcs-q-slider", "value"),
        prevent_initial_call=True,
    )
    def _scan_params(n_clicks, k1_center, q_center):
        if not n_clicks:
            raise PreventUpdate

        try:
            bif_q = requests.post(
                f"{API}/api/v1/frac-chem-sprott/bifurcation",
                json={
                    "scan_param": "q",
                    "start": 0.55,
                    "stop": 1.0,
                    "num": 60,
                    "observable": "x",
                    "params": {"k1": k1_center},
                    "t_max": 60.0,
                    "dt": 0.002,
                    "t_transient": 20.0,
                    "sample_stride": 10,
                },
                timeout=300,
            ).json()

            bif_k1 = requests.post(
                f"{API}/api/v1/frac-chem-sprott/bifurcation",
                json={
                    "scan_param": "k1",
                    "start": 0.0,
                    "stop": 0.07,
                    "num": 60,
                    "observable": "x",
                    "params": {"q": q_center},
                    "t_max": 60.0,
                    "dt": 0.002,
                    "t_transient": 20.0,
                    "sample_stride": 10,
                },
                timeout=300,
            ).json()

            comp = requests.post(
                f"{API}/api/v1/frac-chem-sprott/complexity-grid",
                json={"q_steps": 10, "k1_steps": 10},
                timeout=300,
            ).json()
        except Exception as exc:  # noqa: BLE001
            empty = go.Figure()
            empty.update_layout(title=f"Scan failed: {exc}")
            return empty, empty, empty, empty

        def make_bif_fig(bif, x_label):
            x_vals = bif.get("param_values", [])
            y_samples = bif.get("observable_samples", [])
            xs: list[float] = []
            ys: list[float] = []
            for pv, ysamples in zip(x_vals, y_samples):
                xs.extend([pv] * len(ysamples))
                ys.extend(ysamples)
            fig = go.Figure(
                data=[
                    go.Scattergl(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(size=2),
                    )
                ]
            )
            fig.update_layout(**PLOTLY_DARK_LAYOUT)
            fig.update_layout(
                title=f"Bifurcation vs {x_label}",
                xaxis_title=x_label,
                yaxis_title="observable (x)",
            )
            return fig

        fig_bif_q = make_bif_fig(bif_q, "q")
        fig_bif_k1 = make_bif_fig(bif_k1, "k1")

        q_vals = comp.get("q_values", [])
        k1_vals = comp.get("k1_values", [])
        se = comp.get("se_grid", [])
        c0 = comp.get("c0_grid", [])

        fig_se = go.Figure(
            data=[
                go.Heatmap(
                    x=q_vals,
                    y=k1_vals,
                    z=se,
                    colorbar=dict(title="SE"),
                )
            ]
        )
        fig_se.update_layout(title="Spectral entropy vs (q, k1)", xaxis_title="q", yaxis_title="k1")

        fig_c0 = go.Figure(
            data=[
                go.Heatmap(
                    x=q_vals,
                    y=k1_vals,
                    z=c0,
                    colorbar=dict(title="C0"),
                )
            ]
        )
        fig_c0.update_layout(title="C0 complexity vs (q, k1)", xaxis_title="q", yaxis_title="k1")

        return fig_bif_q, fig_bif_k1, fig_se, fig_c0


register_page(
    "frac_chem_sprott",
    name="Frac Chem Sprott",
    path="/frac-chem-sprott",
    layout=layout,
    register_callbacks=register_callbacks,
    order=220,
)
