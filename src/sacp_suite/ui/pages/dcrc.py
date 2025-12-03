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
                    html.H3("Distributed Chaotic Reservoir Computing", className="section-title"),
                    html.Div(
                        [
                            html.Label("Reservoirs"),
                            dcc.Input(id="dcrc_n", type="number", value=3, step=1),
                            html.Label("Coupling"),
                            dcc.Input(id="dcrc_coupling", type="number", value=1.0, step=0.1),
                            html.Label("Timesteps"),
                            dcc.Input(id="dcrc_ts", type="number", value=1000, step=100),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Run DCRC", id="dcrc_run", n_clicks=0, className="primary-btn"),
                    html.Div(id="dcrc_shared_note", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="dcrc_traj"), type="dot"),
                    dcc.Loading(dcc.Graph(id="dcrc_network"), type="dot"),
                    html.Div(id="dcrc_metrics", style={"paddingTop": "6px"}),
                ],
                className="card",
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("dcrc_traj", "figure"),
        Output("dcrc_network", "figure"),
        Output("dcrc_metrics", "children"),
        Output("dcrc_shared_note", "children"),
        Input("dcrc_run", "n_clicks"),
        State("dcrc_n", "value"),
        State("dcrc_coupling", "value"),
        State("dcrc_ts", "value"),
        prevent_initial_call=True,
    )
    def run_dcrc(n_clicks: int, n_res: int, coupling: float, timesteps: int):
        if not n_clicks:
            raise PreventUpdate
        payload = {"num_reservoirs": n_res, "coupling": coupling, "timesteps": timesteps}
        resp = requests.post(f"{API}/dcrc/run", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        trajectories = data.get("trajectories", [])
        fig_traj = go.Figure()
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        for i, traj in enumerate(trajectories):
            color = colors[i % len(colors)]
            fig_traj.add_trace(
                go.Scatter3d(
                    x=traj["x"],
                    y=traj["y"],
                    z=traj["z"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"Reservoir {i + 1}",
                )
            )
        fig_traj.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig_net = go.Figure()
        theta = np.linspace(0, 2 * np.pi, n_res, endpoint=False)
        node_x = np.cos(theta)
        node_y = np.sin(theta)
        for i in range(n_res):
            for j in range(i + 1, n_res):
                fig_net.add_trace(
                    go.Scatter(
                        x=[node_x[i], node_x[j]],
                        y=[node_y[i], node_y[j]],
                        mode="lines",
                        line=dict(color="gray", width=coupling),
                        showlegend=False,
                    )
                )
        fig_net.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=16, color="red"),
                text=[f"R{i+1}" for i in range(n_res)],
                textposition="middle center",
                name="Reservoirs",
            )
        )
        fig_net.update_layout(
            height=360,
            title="DCRC Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
        )
        shared = get_shared_traj()
        note = ""
        if shared is not None and getattr(shared, "size", 0):
            fig_traj.add_trace(
                go.Scatter3d(
                    x=shared[:, 0],
                    y=shared[:, 1],
                    z=shared[:, 2] if shared.shape[1] > 2 else np.zeros(shared.shape[0]),
                    mode="lines",
                    line=dict(color="black", width=3),
                    name="Shared dataset",
                )
            )
            note = "Shared dataset plotted in black."
        metrics = data.get("metrics", {})
        pred_stats = data.get("pred_samples", [])
        return (
            fig_traj,
            fig_net,
            f"avg_pred={metrics.get('avg_pred', 0):.3f}, std_pred={metrics.get('std_pred', 0):.3f}, samples={len(pred_stats)}",
            note,
        )


register_page(
    "dcrc",
    name="DCRC",
    path="/dcrc",
    layout=layout,
    register_callbacks=register_callbacks,
    order=60,
)
