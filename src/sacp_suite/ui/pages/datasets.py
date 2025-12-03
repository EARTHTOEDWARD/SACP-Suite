from __future__ import annotations

import time
import numpy as np
import requests
import dash
from dash import Input, Output, State, dcc, html
import plotly.graph_objects as go

from sacp_suite.ui.pages.common import API_BASE, set_shared_traj
from sacp_suite.ui.pages import register_page

API = API_BASE


def _make_table(rows: list[dict]) -> html.Table:
    if not rows:
        return html.Table()
    cols = list(rows[0].keys())
    header = html.Tr([html.Th(c) for c in cols])
    body = [html.Tr([html.Td(r.get(c, "")) for c in cols]) for r in rows[:10]]
    return html.Table([header] + body, style={"borderCollapse": "collapse", "width": "100%", "border": "1px solid #e2e8f0"})


def layout():
    return html.Div(
        [
            html.H3("Datasets — Strange Attractors"),
            html.P("Bundled examples from the legacy SACP datasets folder."),
            dcc.Store(id="dataset_refresh"),
            html.Button("Load datasets", id="datasets_load", n_clicks=0),
            dcc.Dropdown(id="dataset_select", placeholder="Select dataset", style={"marginTop": "10px"}),
            html.Div(
                [
                    html.Button("Preview dataset", id="dataset_preview_btn", n_clicks=0, style={"marginTop": "10px"}),
                    html.Button(
                        "Send dataset",
                        id="dataset_send_btn",
                        n_clicks=0,
                        style={"marginTop": "10px", "marginLeft": "8px"},
                    ),
                    dcc.Dropdown(
                        id="dataset_send_target",
                        options=[
                            {"label": "Attractorhedron", "value": "attr"},
                            {"label": "Simulator", "value": "sim"},
                            {"label": "Fractal LLM", "value": "fractalllm"},
                            {"label": "Cognition", "value": "cog"},
                            {"label": "DCRC", "value": "dcrc"},
                            {"label": "Bioelectric", "value": "bcp"},
                            {"label": "ABTC", "value": "abtc"},
                        ],
                        value="attr",
                        clearable=False,
                        style={"width": "220px", "marginLeft": "8px"},
                    ),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
            html.Div(id="dataset_meta", style={"paddingTop": "10px"}),
            dcc.Loading(dcc.Graph(id="dataset_plot"), type="dot"),
            html.Div(id="dataset_table"),
            html.Hr(),
            html.H4("Upload your own dataset (CSV)"),
            html.Div(
                [
                    html.Label("Name"),
                    dcc.Input(id="dataset_upload_name", placeholder="My dataset", style={"width": "100%"}),
                    html.Label("Description", style={"marginTop": "6px"}),
                    dcc.Input(id="dataset_upload_desc", placeholder="Optional", style={"width": "100%"}),
                    dcc.Upload(
                        id="dataset_upload_file",
                        children=html.Div(["Drag and drop or ", html.B("select a CSV")]),
                        style={
                            "width": "100%",
                            "height": "80px",
                            "lineHeight": "80px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "4px",
                            "textAlign": "center",
                            "margin": "10px 0",
                            "background": "#f8fafc",
                        },
                        multiple=False,
                    ),
                    html.Button("Upload dataset", id="dataset_upload_btn", n_clicks=0),
                    html.Div(id="dataset_upload_status", style={"paddingTop": "6px", "fontWeight": "600"}),
                ]
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("dataset_select", "options"),
        Input("datasets_load", "n_clicks"),
        Input("dataset_refresh", "data"),
        prevent_initial_call=True,
    )
    def load_datasets(n_clicks: int, refresh: str | None):
        resp = requests.get(f"{API}/datasets", timeout=30)
        resp.raise_for_status()
        datasets = resp.json().get("datasets", [])
        return [{"label": f"{d['name']} — {d['description']}", "value": d["id"]} for d in datasets]

    @app.callback(
        Output("dataset_plot", "figure"),
        Output("dataset_table", "children"),
        Output("dataset_meta", "children", allow_duplicate=True),
        Output("shared_traj", "data"),
        Input("dataset_preview_btn", "n_clicks"),
        State("dataset_select", "value"),
        prevent_initial_call=True,
    )
    def preview_dataset(n_clicks: int, dataset_id: str | None):
        if not dataset_id:
            return go.Figure(), "", "Select a dataset first.", dash.no_update
        resp = requests.post(f"{API}/datasets/preview", json={"dataset_id": dataset_id, "limit": 2000}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        cols = data.get("columns", [])
        fig = go.Figure()
        traj_store = None
        if rows and len(cols) >= 3:
            arr = np.array([[r.get(cols[1], 0), r.get(cols[2], 0)] for r in rows], dtype=float)
            fig.add_trace(go.Scatter(x=arr[:, 0], y=arr[:, 1], mode="markers", marker=dict(size=4)))
            fig.update_layout(height=360, xaxis_title=cols[1], yaxis_title=cols[2])
            if len(cols) >= 4:
                traj_store = [[r.get(cols[1], 0), r.get(cols[2], 0), r.get(cols[3], 0)] for r in rows]
            else:
                traj_store = [[r.get(cols[0], 0), r.get(cols[1], 0), r.get(cols[2], 0)] for r in rows]
        table = _make_table(rows)
        hint = ""
        dsid_lower = (data.get("id", "") or "").lower()
        if "lorenz" in dsid_lower:
            hint = "Try Simulator and Attractorhedron."
        elif "rossler" in dsid_lower:
            hint = "Try Simulator (Lorenz slots) and Attractorhedron."
        elif "hr_" in dsid_lower:
            hint = "Use Simulator/Attractorhedron; good for chaos metrics."
        elif "embed" in dsid_lower:
            hint = "Try Fractal LLM or Bioelectric (time-series)."
        meta = f"{data.get('name', dataset_id)} — {data.get('description', '')} (showing {len(rows)} rows; est {data.get('total_estimate', 'n/a')}) {hint}"
        return fig, table, meta, traj_store

    @app.callback(
        Output("dataset_upload_status", "children"),
        Output("dataset_refresh", "data"),
        Input("dataset_upload_btn", "n_clicks"),
        State("dataset_upload_name", "value"),
        State("dataset_upload_desc", "value"),
        State("dataset_upload_file", "contents"),
        prevent_initial_call=True,
    )
    def upload_dataset(n_clicks: int, name: str | None, desc: str | None, contents: str | None):
        if not contents:
            return "Upload a CSV first.", dash.no_update
        payload = {
            "name": name or "upload",
            "description": desc or "",
            "contents_b64": contents,
            "columns": [],
        }
        resp = requests.post(f"{API}/datasets/upload", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return f"Uploaded '{data.get('name')}' (id: {data.get('id')})", str(time.time())

    @app.callback(
        Output("dataset_meta", "children", allow_duplicate=True),
        Input("dataset_send_btn", "n_clicks"),
        State("dataset_send_target", "value"),
        State("shared_traj", "data"),
        prevent_initial_call=True,
    )
    def send_dataset(n_clicks: int, target: str, traj):
        if not traj:
            return "Preview a dataset first, then send."
        set_shared_traj(np.array(traj, dtype=float))
        hint = {
            "attr": "Open Attractorhedron and click Build.",
            "sim": "Open Simulator and click 'Show shared trajectory'.",
            "fractalllm": "Open Fractal LLM and reuse the embedded plot; future: drive reservoir.",
            "cog": "Open Cognition to drive memory/discriminability with shared data soon.",
            "dcrc": "Open DCRC to visualize alongside synthetic trajectories.",
            "bcp": "Open Bioelectric to section/inspect.",
            "abtc": "Open ABTC to plot trajectories with dt/steps.",
        }.get(target, "Dataset stored.")
        return f"Dataset sent (target: {target}). {hint}"


register_page(
    "datasets",
    name="Datasets",
    path="/datasets",
    layout=layout,
    register_callbacks=register_callbacks,
    order=100,
)
