from __future__ import annotations

import json
from typing import Any, Dict

import dash
import requests
from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from sacp_suite.ui.pages.common import API_BASE, parse_numeric_list
from sacp_suite.ui.pages import register_page

API = API_BASE


def layout():
    preview_table = dash_table.DataTable(
        id="preview_table",
        data=[],
        columns=[{"name": c, "id": c} for c in ["t", "x", "y", "z", "u1"]],
        page_size=10,
        style_table={"maxHeight": "240px", "overflowY": "auto"},
        style_cell={"fontFamily": "JetBrains Mono, Menlo, monospace", "textAlign": "left"},
    )
    dataset_table = dash_table.DataTable(
        id="dataset_preview_table",
        data=[],
        columns=[{"name": c, "id": c} for c in ["t", "x", "y", "z", "u1"]],
        page_size=6,
        style_table={"maxHeight": "200px", "overflowY": "auto"},
        style_cell={"fontFamily": "JetBrains Mono, Menlo, monospace", "textAlign": "left"},
    )

    return html.Div(
        [
            html.Div(
                [
                    html.H1("What do you want to understand or control?", className="landing-title"),
                    html.P(
                        "Describe your system or study in natural language. "
                        "We’ll translate it into a structured plan the Suite can execute."
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong("Example 1"),
                                    html.Span(": Fit a chaotic model to my EEG recordings and detect transitions"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Strong("Example 2"),
                                    html.Span(": Monitor my control system in real time and flag unstable behavior"),
                                ]
                            ),
                        ],
                        className="prompt-examples",
                    ),
                    dcc.Textarea(
                        id="hero_prompt_input",
                        placeholder="Example: Fit a chaotic model to my EEG recordings and detect transitions",
                        style={"width": "100%", "minHeight": "120px"},
                    ),
                    html.Div(
                        [
                            html.Button("Ask the Suite", id="prompt_submit_btn", n_clicks=0, className="primary-btn"),
                            dcc.Loading(html.Div(id="prompt_status", className="status-text"), type="dot"),
                        ],
                        className="prompt-actions",
                    ),
                    html.Div(id="task_spec_summary", className="task-summary"),
                ],
                className="card landing-card",
            ),
            html.Div(
                [
                    html.H3("Set the stage"),
                    html.P("Quick toggles that the assistant can override if your prompt says otherwise."),
                    html.Label("Do you already have data?", htmlFor="data_status_toggle"),
                    dcc.RadioItems(
                        id="data_status_toggle",
                        options=[
                            {"label": "I already have data", "value": "have_data"},
                            {"label": "I don’t have data yet", "value": "need_dataset"},
                        ],
                        value="have_data",
                        className="inline-radio",
                    ),
                    html.Label("How do you want to work?", htmlFor="mode_toggle"),
                    dcc.RadioItems(
                        id="mode_toggle",
                        options=[
                            {"label": "Historical / static data", "value": "batch"},
                            {"label": "Real-time / streaming", "value": "realtime"},
                        ],
                        value="batch",
                        className="inline-radio",
                    ),
                    html.Button("Continue", id="continue_flow_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="continue_hint", className="status-text"),
                ],
                className="card landing-card",
            ),
            html.Div(
                [
                    html.H3("Flow A · I already have data"),
                    html.P("We’ll validate that the files, buckets, or streams you pointed to are ready for SACP analyses."),
                    html.H4("Step 1 — Where is your data?"),
                    dcc.RadioItems(
                        id="data_source_selector",
                        options=[
                            {"label": "File upload (CSV, Parquet, HDF5)", "value": "file"},
                            {"label": "Cloud storage (S3 / GCS / Azure)", "value": "s3"},
                            {"label": "Database (Postgres, MySQL…)", "value": "db"},
                            {"label": "API / live stream (WebSocket, MQTT…)", "value": "api"},
                        ],
                        value="file",
                        className="stacked-radio",
                    ),
                    html.Div(
                        [
                            html.Label("How many sample rows should we pull?"),
                            dcc.Slider(
                                id="preview_rows_slider",
                                min=5,
                                max=40,
                                step=5,
                                value=15,
                                marks={5: "5", 20: "20", 40: "40"},
                            ),
                        ],
                        className="inline-form",
                    ),
                    html.Button("Load a sample", id="load_preview_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="preview_status", className="status-text"),
                    dcc.Loading(preview_table, type="default"),
                    html.Div(id="row_estimate_badge", className="status-text"),
                    html.H4("Step 2 — Map your columns"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Time axis"),
                                    dcc.Dropdown(id="time_column_dropdown", placeholder="Pick the time column"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("State variables"),
                                    dcc.Dropdown(
                                        id="state_columns_dropdown",
                                        multi=True,
                                        placeholder="Pick one or more columns",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Control inputs (optional)"),
                                    dcc.Dropdown(
                                        id="control_columns_dropdown",
                                        multi=True,
                                        placeholder="Select optional inputs",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Group / trial id"),
                                    dcc.Dropdown(id="group_column_dropdown", placeholder="Optional group column"),
                                ]
                            ),
                        ],
                        className="mapping-grid",
                    ),
                    html.H4("Step 3 — Automatic health checks"),
                    html.Div(
                        [
                            html.Label("How many rows are you planning to ingest?"),
                            dcc.Input(id="row_count_input", type="number", value=2000, style={"width": "160px"}),
                        ],
                        className="inline-form",
                    ),
                    html.Button("Run health checks", id="run_validation_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="validation_results", className="validation-box"),
                    html.H4("Step 4 — Batch vs real-time"),
                    html.Div(id="mode_context_text", className="status-text"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("How frequently do points arrive?"),
                                    dcc.Input(
                                        id="arrival_rate_input",
                                        placeholder="e.g. every 10 ms",
                                        style={"width": "160px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("How fast do you need feedback?"),
                                    dcc.Slider(
                                        id="feedback_speed_slider",
                                        min=0.1,
                                        max=10,
                                        step=0.1,
                                        value=5,
                                        marks={0.1: "100 ms", 1: "1 s", 5: "5 s", 10: "10 s"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Alert channels"),
                                    dcc.Checklist(
                                        id="alert_channel_checklist",
                                        options=[
                                            {"label": "Email", "value": "email"},
                                            {"label": "Webhook", "value": "webhook"},
                                            {"label": "Log only", "value": "log"},
                                        ],
                                        value=["log"],
                                        className="inline-check",
                                    ),
                                ]
                            ),
                        ],
                        className="mapping-grid",
                    ),
                    html.Div(id="realtime_config_preview", className="config-preview"),
                ],
                id="flow_have_data",
                className="card landing-card",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.H3("Flow B · I need a dataset"),
                    html.P("Use the dynamical systems builder to spin up a dataset that behaves like your intent."),
                    html.H4("Step 1 — Quick intent clarification"),
                    dcc.Dropdown(
                        id="system_template_dropdown",
                        options=[
                            {"label": "Canonical chaotic (Lorenz, Rossler…)", "value": "lorenz"},
                            {"label": "Biological / bioelectric", "value": "bio"},
                            {"label": "Control / engineering systems", "value": "control"},
                            {"label": "Custom / unsure", "value": "custom"},
                        ],
                        value="lorenz",
                    ),
                    html.H4("Step 2 — Dataset builder"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Experiment length"),
                                    dcc.RadioItems(
                                        id="experiment_length_radio",
                                        options=[
                                            {"label": "Short (≈10 s)", "value": "short"},
                                            {"label": "Medium (≈1 min)", "value": "medium"},
                                            {"label": "Long (training)", "value": "long"},
                                        ],
                                        value="medium",
                                        className="stacked-radio",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Sampling rate"),
                                    dcc.RadioItems(
                                        id="sampling_rate_radio",
                                        options=[
                                            {"label": "Slow (10–50 Hz)", "value": "slow"},
                                            {"label": "Medium (100–500 Hz)", "value": "medium"},
                                            {"label": "Fast (1 kHz+)", "value": "fast"},
                                        ],
                                        value="medium",
                                        className="stacked-radio",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Noise level"),
                                    dcc.Slider(
                                        id="noise_level_slider",
                                        min=0.0,
                                        max=0.5,
                                        step=0.05,
                                        value=0.1,
                                        marks={0.0: "0", 0.25: "0.25", 0.5: "0.5"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Trials"),
                                    dcc.Input(id="trials_input", type="number", value=3, style={"width": "120px"}),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Parameter variations"),
                                    dcc.Input(id="variations_input", type="number", value=1, style={"width": "120px"}),
                                ]
                            ),
                        ],
                        className="mapping-grid",
                    ),
                    html.Button("Generate dataset", id="dataset_generate_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="dataset_status", className="status-text"),
                    dcc.Loading(dataset_table, type="default"),
                    html.Div(id="dataset_summary", className="config-preview"),
                ],
                id="flow_need_dataset",
                className="card landing-card",
                style={"display": "none"},
            ),
            html.Div(
                [
                    html.H4("How much data do I need?"),
                    html.P(
                        "SACP never hard-blocks you, but different modules have practical ranges. "
                        "Use this cheat sheet while exploring."
                    ),
                    html.Ul(
                        [
                            html.Li("A few hundred points → quick plots and intuition."),
                            html.Li("A few thousand per regime → attractor geometry + chaos indicators."),
                            html.Li("Tens of thousands → predictive models and fine-grained stats."),
                        ]
                    ),
                    html.P(
                        "Real-time monitoring keeps a sliding window (seconds to minutes) so you only need enough "
                        "history to detect transitions."
                    ),
                    html.P(
                        "If your system changes slowly, plan for longer sessions or multiple windows — the wizard "
                        "will highlight which modules might be unreliable when samples are scarce."
                    ),
                ],
                className="card landing-card info-card",
            ),
        ],
        className="landing-grid",
    )


def register_callbacks(app):
    @app.callback(
        Output("task_spec_store", "data"),
        Output("prompt_status", "children"),
        Input("prompt_submit_btn", "n_clicks"),
        State("hero_prompt_input", "value"),
        prevent_initial_call=True,
    )
    def parse_task_spec(n_clicks: int, prompt: str | None):
        if not n_clicks:
            raise PreventUpdate
        payload = {"prompt": prompt or ""}
        try:
            resp = requests.post(f"{API}/api/parse-task", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            confidence = data.get("confidence")
            message = "Intent captured."
            if confidence is not None:
                message = f"Intent captured · confidence {confidence:.0%}"
            return data, message
        except Exception as exc:  # pragma: no cover - network failures
            return dash.no_update, f"Could not parse intent: {exc}"

    @app.callback(
        Output("data_status_toggle", "value"),
        Output("mode_toggle", "value"),
        Output("task_spec_summary", "children"),
        Input("task_spec_store", "data"),
    )
    def reflect_task_spec(spec: Dict[str, Any] | None):
        if not spec:
            return dash.no_update, dash.no_update, ""
        goal = spec.get("goal", "").strip() or "—"
        confidence = spec.get("confidence")
        has_data = spec.get("has_data")
        summary = html.Div(
            [
                html.Div(f"Goal: {goal}", className="summary-line"),
                html.Div(
                    f"Data: {'provided' if has_data else 'need a dataset'} · Mode: {spec.get('mode', 'batch')}",
                    className="summary-line",
                ),
                html.Div(
                    f"Latency tolerance: {spec.get('latency_tolerance_seconds', '—')} s · "
                    f"Estimated size: {spec.get('estimated_data_size', '—')}",
                    className="summary-line",
                ),
                html.Div(
                    f"Confidence: {confidence:.0%}" if confidence is not None else "Confidence: —",
                    className="summary-line",
                ),
                html.Pre(json.dumps(spec, indent=2), className="json-preview"),
            ],
            className="task-summary-box",
        )
        return spec.get("data_status", "have_data"), spec.get("mode", "batch"), summary

    @app.callback(
        Output("landing_stage", "data"),
        Output("continue_hint", "children"),
        Input("continue_flow_btn", "n_clicks"),
        State("data_status_toggle", "value"),
        State("landing_stage", "data"),
        prevent_initial_call=True,
    )
    def advance_to_flow(n_clicks: int, data_status: str, stage: str):
        if not n_clicks:
            raise PreventUpdate
        message = (
            "Jumping into Flow A (ingest + validation)."
            if data_status == "have_data"
            else "Jumping into Flow B (dataset builder)."
        )
        return "flow", message

    @app.callback(
        Output("flow_have_data", "style"),
        Output("flow_need_dataset", "style"),
        Input("landing_stage", "data"),
        Input("data_status_toggle", "value"),
    )
    def toggle_flows(stage: str, data_status: str):
        hidden = {"display": "none"}
        visible = {"display": "block"}
        if stage != "flow":
            return hidden, hidden
        if data_status == "have_data":
            return visible, hidden
        return hidden, visible

    @app.callback(Output("mode_context_text", "children"), Input("mode_toggle", "value"))
    def update_mode_context(mode: str):
        if mode == "realtime":
            return (
                "Real-time monitoring keeps a sliding window, pushes alerts when thresholds trip, "
                "and expects your stream frequency + notification preferences."
            )
        return "Batch mode stores the ingest spec and lets you run attractor, Lyapunov, or modeling jobs immediately."

    @app.callback(
        Output("ingestion_preview_store", "data"),
        Output("preview_status", "children"),
        Input("load_preview_btn", "n_clicks"),
        State("data_source_selector", "value"),
        State("preview_rows_slider", "value"),
        prevent_initial_call=True,
    )
    def load_preview_sample(n_clicks: int, source: str, rows: int):
        if not n_clicks:
            raise PreventUpdate
        payload = {"source": source, "sample_size": rows, "modality": "time_series"}
        try:
            resp = requests.post(f"{API}/api/ingest/preview", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pulled = len(data.get("rows", []))
            return data, f"Loaded {pulled} preview rows from {source.upper()}."
        except Exception as exc:  # pragma: no cover - network failures
            return dash.no_update, f"Could not load preview: {exc}"

    @app.callback(
        Output("preview_table", "data"),
        Output("preview_table", "columns"),
        Output("row_estimate_badge", "children"),
        Input("ingestion_preview_store", "data"),
    )
    def populate_preview_table(data: Dict[str, Any] | None):
        default_columns = [{"name": c, "id": c} for c in ["t", "x", "y", "z", "u1"]]
        if not data:
            return [], default_columns, ""
        columns = data.get("columns")
        rows = data.get("rows", [])
        if not columns and rows:
            columns = list(rows[0].keys())
        if not columns:
            columns = [c["id"] for c in default_columns]
        formatted_columns = [{"name": col, "id": col} for col in columns]
        estimate = data.get("row_estimate")
        estimate_text = f"Estimated total rows: ~{estimate:,}" if isinstance(estimate, int) else ""
        return rows, formatted_columns, estimate_text

    @app.callback(
        Output("time_column_dropdown", "options"),
        Output("state_columns_dropdown", "options"),
        Output("control_columns_dropdown", "options"),
        Output("group_column_dropdown", "options"),
        Input("ingestion_preview_store", "data"),
    )
    def update_column_options(data: Dict[str, Any] | None):
        if not data:
            return [], [], [], []
        columns = data.get("columns")
        rows = data.get("rows", [])
        if not columns and rows:
            columns = list(rows[0].keys())
        options = [{"label": col, "value": col} for col in (columns or [])]
        return options, options, options, options

    @app.callback(
        Output("validation_results", "children"),
        Input("run_validation_btn", "n_clicks"),
        State("time_column_dropdown", "value"),
        State("state_columns_dropdown", "value"),
        State("control_columns_dropdown", "value"),
        State("group_column_dropdown", "value"),
        State("row_count_input", "value"),
        State("mode_toggle", "value"),
        State("alert_channel_checklist", "value"),
        prevent_initial_call=True,
    )
    def run_health_checks(
        n_clicks: int,
        time_col: str | None,
        state_cols: list[str] | None,
        control_cols: list[str] | None,
        group_col: str | None,
        row_count: int | None,
        mode: str,
        alert_channels: list[str] | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not time_col or not state_cols:
            return html.Div(
                "Select at least a time column and one state variable before running checks.",
                className="status-text",
            )
        payload = {
            "mapping": {
                "time_col": time_col,
                "state_cols": state_cols or [],
                "control_cols": control_cols or [],
                "group_col": group_col,
            },
            "row_count": row_count or 0,
            "needs_alerts": mode == "realtime" and bool(alert_channels),
        }
        try:
            resp = requests.post(f"{API}/api/ingest/validate", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - network failures
            return html.Div(f"Validation failed: {exc}", className="status-text error")

        checks = data.get("format_checks", [])
        suff = data.get("sufficiency", {})
        check_items = [
            html.Li(f"{c.get('label')}: {c.get('status', '').upper()}", className=f"check-{c.get('status', 'pass')}")
            for c in checks
        ]
        suff_block = html.Div(
            [
                html.Strong(f"Sufficiency → {suff.get('label', 'Unknown')}"),
                html.Div(suff.get("message", "")),
                html.Small(f"Rows reported: {suff.get('row_count', row_count or 0):,}"),
            ],
            className="sufficiency-box",
        )
        return html.Div([html.Ul(check_items), suff_block], className="validation-summary")

    @app.callback(
        Output("realtime_config_preview", "children"),
        Input("mode_toggle", "value"),
        Input("arrival_rate_input", "value"),
        Input("feedback_speed_slider", "value"),
        Input("alert_channel_checklist", "value"),
    )
    def update_realtime_preview(mode: str, arrival_rate: str | None, feedback_speed: float, channels: list[str] | None):
        if mode != "realtime":
            return "Batch mode selected — we’ll process what you uploaded and hand you analyses immediately."
        config = {
            "mode": "realtime",
            "arrival_hint": arrival_rate or "unspecified",
            "update_interval_seconds": round((feedback_speed or 5), 2),
            "window_length_seconds": max(5, min(180, int((feedback_speed or 5) * 6))),
            "alert_channels": channels or [],
        }
        return html.Pre(json.dumps(config, indent=2), className="json-preview")

    @app.callback(
        Output("dataset_builder_store", "data"),
        Output("dataset_status", "children"),
        Input("dataset_generate_btn", "n_clicks"),
        State("system_template_dropdown", "value"),
        State("experiment_length_radio", "value"),
        State("sampling_rate_radio", "value"),
        State("noise_level_slider", "value"),
        State("trials_input", "value"),
        State("variations_input", "value"),
        prevent_initial_call=True,
    )
    def generate_dataset(
        n_clicks: int,
        system: str,
        duration: str,
        sampling_rate: str,
        noise_level: float,
        trials: int,
        variations: int,
    ):
        if not n_clicks:
            raise PreventUpdate
        payload = {
            "system": system,
            "duration": duration,
            "sampling_rate": sampling_rate,
            "noise_level": noise_level,
            "trials": trials or 1,
            "variations": variations or 0,
        }
        try:
            resp = requests.post(f"{API}/api/dataset/generate", json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("row_estimate")
            message = f"Synthetic dataset ready (~{rows:,} rows)." if isinstance(rows, int) else "Synthetic dataset ready."
            return data, message
        except Exception as exc:  # pragma: no cover - network failures
            return dash.no_update, f"Dataset generation failed: {exc}"

    @app.callback(
        Output("dataset_preview_table", "data"),
        Output("dataset_preview_table", "columns"),
        Output("dataset_summary", "children"),
        Input("dataset_builder_store", "data"),
    )
    def populate_dataset_preview(data: Dict[str, Any] | None):
        default_columns = [{"name": c, "id": c} for c in ["t", "x", "y", "z", "u1"]]
        if not data:
            return [], default_columns, ""
        preview = data.get("preview", {})
        rows = preview.get("rows", [])
        columns = preview.get("columns")
        if not columns and rows:
            columns = list(rows[0].keys())
        formatted_columns = [{"name": col, "id": col} for col in (columns or [])] or default_columns
        summary = html.Div(
            [
                html.Div(f"Dataset id: {data.get('id', 'pending')}"),
                html.Div(
                    f"Estimated rows: {data.get('row_estimate', '—'):,}"
                    if isinstance(data.get("row_estimate"), int)
                    else "Estimated rows: —"
                ),
                html.Div(data.get("message", "")),
                html.Pre(json.dumps(data.get("ingest_mapping", {}), indent=2), className="json-preview"),
            ],
            className="task-summary-box",
        )
        return rows, formatted_columns, summary


register_page(
    "home",
    name="Home",
    path="/",
    layout=layout,
    register_callbacks=register_callbacks,
    order=0,
)
