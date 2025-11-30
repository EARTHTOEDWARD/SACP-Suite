"""Dash multi-page UI for SACP Suite."""

from __future__ import annotations

import os
import base64
import io
import json
import time
from typing import Tuple

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from sacp_suite.modules.fractalhedron import (
    build_fractalhedron_k,
    build_symbolic_sequence,
    fractal_face_flags,
)

API = f"http://{os.getenv('SACP_BIND', '127.0.0.1')}:{os.getenv('SACP_PORT', '8000')}"

DEFAULT_COG_ALPHABET = [-1.0, 0.0, 1.0]
DEFAULT_COG_PATTERNS = [
    [1.0, 0.0, -1.0, 0.0],
    [1.0, 1.0, 0.0, -1.0],
    [0.0, -1.0, -1.0, 1.0],
]

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="SACP Suite")
server = app.server

NAV = html.Nav(
    [
        dcc.Link("Home", href="/"),
        html.Span(" | "),
        dcc.Link("Simulator", href="/sim"),
        html.Span(" | "),
        dcc.Link("Sheaf", href="/sheaf"),
        html.Span(" | "),
        dcc.Link("Attractorhedron", href="/attr"),
        html.Span(" | "),
        dcc.Link("Fractal LLM", href="/fractalllm"),
        html.Span(" | "),
        dcc.Link("Cognition", href="/cog"),
        html.Span(" | "),
        dcc.Link("Self Tuning", href="/self-tune"),
        html.Span(" | "),
        dcc.Link("DCRC", href="/dcrc"),
        html.Span(" | "),
        dcc.Link("Bioelectric", href="/bcp"),
        html.Span(" | "),
        dcc.Link("ABTC", href="/abtc"),
        html.Span(" | "),
        dcc.Link("Datasets", href="/datasets"),
    ],
    style={"padding": "8px", "fontFamily": "Segoe UI, Helvetica Neue, sans-serif"},
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.Store(id="shared_traj"),
        dcc.Store(id="task_spec_store"),
        dcc.Store(id="landing_stage", data="prompt"),
        dcc.Store(id="ingestion_preview_store"),
        dcc.Store(id="dataset_builder_store"),
        NAV,
        html.Div(id="page"),
    ],
    style={
        "maxWidth": "1280px",
        "margin": "0 auto",
        "padding": "16px",
    },
)


def page_home():
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


def page_sim():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Simulator — Lorenz 1963", className="section-title"),
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
                    html.Button("Run", id="run", n_clicks=0, className="primary-btn"),
                    html.Button("Show shared trajectory", id="show_shared_traj", n_clicks=0, style={"marginLeft": "8px"}),
                    html.Div(
                        [
                            dcc.Loading(dcc.Graph(id="phase3d"), type="dot"),
                            dcc.Loading(dcc.Graph(id="series"), type="dot"),
                            html.Div(id="lle_out", style={"paddingTop": "6px", "fontWeight": "600"}),
                        ],
                        className="section-grid",
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("Upload your data to compute LLE", className="section-title"),
                    html.P("Upload a CSV with one numeric column (or pick the first column).", style={"color": "#4b5563"}),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(["Drag and drop or ", html.B("select a CSV")]),
                        className="upload-box",
                        multiple=False,
                    ),
                    html.Button("Compute LLE from file", id="lle_from_file_btn", n_clicks=0, className="primary-btn"),
                    html.Div(id="lle_file_out", style={"paddingTop": "6px", "fontWeight": "600"}),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def page_sheaf():
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
                                style={"flex": 1, "display": "grid", "gridTemplateColumns": "repeat(2, minmax(0,1fr))", "gap": "6px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "padding": "6px"},
                    ),
                    html.Button("Run sheaf sweep", id="sheaf-run", n_clicks=0, className="primary-btn"),
                    html.Div(id="sheaf-summary", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="sheaf-lambda-fig"), type="dot"),
                    dcc.Loading(dcc.Graph(id="sheaf-clc-fig"), type="dot"),
                    html.H4("Sections / obstructions"),
                    dash_table.DataTable(
                        id="sheaf-sections-table",
                        columns=[
                            {"name": "start", "id": "start"},
                            {"name": "end", "id": "end"},
                            {"name": "class", "id": "class_label"},
                            {"name": "persist", "id": "persist"},
                        ],
                        data=[],
                        page_size=10,
                        style_table={"maxHeight": "240px", "overflowY": "auto"},
                        style_cell={"fontFamily": "JetBrains Mono, Menlo, monospace", "textAlign": "left"},
                    ),
                    html.Div(id="sheaf-obstructions", style={"paddingTop": "6px"}),
                ],
                className="card",
            )
        ]
    )


def page_attr():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Attractorhedron — data-driven operator", className="section-title"),
                    html.P("Use the last Lorenz run or sent dataset to build an operator; inspect |lambda2|, gamma, and v2."),
                    html.Button("Build from last sim", id="build_op", n_clicks=0, className="primary-btn"),
                    html.Div(id="op_stats", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="v2_map"), type="dot"),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("FractalHedron₂ — symbolic multifractal", className="section-title"),
                    html.P("Use the latest shared trajectory and choose a coding scheme for the Lorenz section hits."),
                    html.Div(
                        [
                            html.Label("Coding scheme"),
                            dcc.Dropdown(
                                id="fh-coding-spec",
                                options=[
                                    {"label": "Binary lobes (x-sign)", "value": "x_sign"},
                                    {"label": "Quadrants (X/Z plane)", "value": "quadrant_xz"},
                                    {"label": "Radius bins", "value": "radius_bins"},
                                ],
                                value="x_sign",
                                clearable=False,
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Radius bins (comma separated)"),
                                    dcc.Input(
                                        id="fh-radius-bins",
                                        type="text",
                                        placeholder="auto",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                            html.Div(
                                [
                                    html.Label("Radius labels (comma separated)"),
                                    dcc.Input(
                                        id="fh-radius-labels",
                                        type="text",
                                        placeholder="inner,middle,outer",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "10px"},
                    ),
                    html.Button("Run FractalHedron₂", id="fh-run", n_clicks=0, className="primary-btn"),
                    html.Button(
                        "Use cached Attractorhedron hits",
                        id="fh-run-cached",
                        n_clicks=0,
                        style={"marginLeft": "8px"},
                    ),
                    dcc.Loading(dcc.Graph(id="fh-dq-graph"), type="dot"),
                    html.Div(id="fh-summary", style={"paddingTop": "6px"}),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def page_fractalllm():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Fractal LLM Lab — reservoir demo", className="section-title"),
                    html.Div(
                        [
                            html.Label("Reservoir size"),
                            dcc.Slider(id="fllm_res_size", min=100, max=1000, step=50, value=400),
                            html.Label("Coupling (spectral radius)"),
                            dcc.Slider(id="fllm_coupling", min=0.2, max=1.5, step=0.05, value=1.0),
                            html.Label("Input text"),
                            dcc.Textarea(
                                id="fllm_text",
                                value="The strange attractor exhibits chaotic behavior",
                                style={"width": "100%", "height": "80px"},
                            ),
                        ],
                        style={"padding": "6px"},
                    ),
                    html.Button("Run Fractal LLM", id="fllm_run", n_clicks=0, className="primary-btn"),
                    dcc.Loading(dcc.Graph(id="fllm_traj"), type="dot"),
                    dcc.Loading(dcc.Graph(id="fllm_switch"), type="dot"),
                    html.Div(id="fllm_stats", style={"paddingTop": "6px", "fontWeight": "600"}),
                    html.Div(id="fllm_shared_note", style={"paddingTop": "6px", "color": "#4b5563"}),
                ],
                className="card",
            )
        ]
    )


def page_cognition():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Cognition — Memory & Discriminability", className="section-title"),
                    html.P(
                        "Probe how a forced Lorenz-63 responds to random drives and temporal concepts. "
                        "Tune the forcing amplitude, dt, and drive alphabet before running experiments."
                    ),
                    html.Div(
                        [
                            html.Label("Max lag (steps)"),
                            dcc.Slider(
                                id="cog-max-lag",
                                min=5,
                                max=100,
                                step=5,
                                value=50,
                                marks={i: str(i) for i in range(10, 101, 20)},
                            ),
                            html.Label("Drive amplitude (Δrho)"),
                            dcc.Slider(id="cog-amp", min=0.0, max=5.0, step=0.1, value=2.0),
                            html.Label("Base rho"),
                            dcc.Slider(id="cog-base-value", min=20.0, max=40.0, step=0.5, value=28.0),
                            html.Label("dt"),
                            dcc.Slider(
                                id="cog-dt",
                                min=0.001,
                                max=0.05,
                                step=0.001,
                                value=0.01,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"padding": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Trials"),
                                    dcc.Input(id="cog-n-trials", type="number", min=1, step=1, value=4, style={"width": "100%"}),
                                ],
                                style={"flex": 1},
                            ),
                            html.Div(
                                [
                                    html.Label("Steps per trial"),
                                    dcc.Input(
                                        id="cog-n-steps",
                                        type="number",
                                        min=200,
                                        step=100,
                                        value=5000,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                            html.Div(
                                [
                                    html.Label("Burn-in steps"),
                                    dcc.Input(
                                        id="cog-burn-in",
                                        type="number",
                                        min=0,
                                        step=50,
                                        value=500,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "padding": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Input alphabet (comma separated)"),
                            dcc.Input(
                                id="cog-alphabet",
                                type="text",
                                value="-1, 0, 1",
                                style={"width": "100%"},
                            ),
                        ],
                        style={"padding": "6px"},
                    ),
                    html.Button("Run memory experiment", id="cog-run-memory", n_clicks=0, className="primary-btn"),
                    dcc.Loading(dcc.Graph(id="cog-memory-graph"), type="dot"),
                    dcc.Loading(dcc.Graph(id="cog-memory-sample"), type="dot"),
                    html.Div(id="cog-clc-summary", style={"paddingTop": "8px", "fontWeight": "600"}),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("Discriminability D(K, T)", className="section-title"),
                    html.P("Compare fixed symbol patterns driven into the same forced simulator (comma-separated values)."),
                    dcc.Textarea(
                        id="cog-patterns",
                        value="\n".join([",".join(str(v) for v in pattern) for pattern in DEFAULT_COG_PATTERNS]),
                        style={"width": "100%", "height": "120px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Trials per concept"),
                                    dcc.Input(
                                        id="cog-disc-trials",
                                        type="number",
                                        min=5,
                                        step=5,
                                        value=100,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                            html.Div(
                                [
                                    html.Label("Burn-in steps"),
                                    dcc.Input(
                                        id="cog-disc-burn",
                                        type="number",
                                        min=0,
                                        step=50,
                                        value=500,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": 1},
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "paddingTop": "6px"},
                    ),
                    html.Button("Run discriminability experiment", id="cog-run-disc", n_clicks=0, className="primary-btn"),
                    dcc.Loading(html.Div(id="cog-disc-summary", style={"marginTop": "12px"}), type="dot"),
                ],
                className="card",
                style={"marginTop": "16px"},
            ),
        ]
    )


def page_dcrc():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("DCRC — Digital Chaotic Reservoir Computing", className="section-title"),
                    html.Div(
                        [
                            html.Label("Number of reservoirs"),
                            dcc.Slider(id="dcrc_n", min=1, max=10, step=1, value=3),
                            html.Label("Coupling strength"),
                            dcc.Slider(id="dcrc_coupling", min=0.1, max=5.0, step=0.1, value=1.0),
                            html.Label("Timesteps"),
                            dcc.Slider(id="dcrc_ts", min=200, max=2000, step=100, value=1000),
                        ],
                        style={"padding": "6px"},
                    ),
                    html.Button("Run DCRC", id="dcrc_run", n_clicks=0, className="primary-btn"),
                    dcc.Loading(dcc.Graph(id="dcrc_traj"), type="dot"),
                    dcc.Loading(dcc.Graph(id="dcrc_network"), type="dot"),
                    html.Div(id="dcrc_metrics", style={"paddingTop": "6px", "fontWeight": "600"}),
                    html.Div(id="dcrc_shared_note", style={"paddingTop": "6px", "color": "#4b5563"}),
                ],
                className="card",
            )
        ]
    )


def page_self_tune():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Self-Tuned Lorenz", className="section-title"),
                    html.P("Run the self-tuning loop and inspect λ₁(t), spectral radius, and the trajectory."),
                    html.Label("Steps"),
                    dcc.Slider(id="self-tune-steps", min=500, max=5000, step=100, value=2000),
                    html.Button("Run Self-Tuner", id="self-tune-run", n_clicks=0, className="primary-btn"),
                    html.Div(id="self-tune-session-id", style={"display": "none"}),
                    dcc.Interval(id="self-tune-timer", interval=2000, disabled=True),
                    html.Div(id="self-tune-summary", style={"paddingTop": "6px"}),
                    dcc.Loading(dcc.Graph(id="self-tune-lambda"), type="dot"),
                    dcc.Loading(dcc.Graph(id="self-tune-sr"), type="dot"),
                    dcc.Loading(dcc.Graph(id="self-tune-attractor"), type="dot"),
                ],
                className="card",
            )
        ]
    )


def page_bcp():
    return html.Div(
        [
            html.H3("Bioelectric Control Panel — sectioning stub"),
            html.P("Send sample (x,z) points to the API sectioner; displays count and scatter."),
            html.Button("Build sample section", id="bcp_run", n_clicks=0, className="primary-btn"),
            dcc.Loading(dcc.Graph(id="bcp_plot"), type="dot"),
            html.Div(id="bcp_stats", style={"paddingTop": "6px", "fontWeight": "600"}),
            html.Div(id="bcp_shared_note", style={"paddingTop": "6px", "color": "#4b5563"}),
        ]
    )


def page_abtc():
    return html.Div(
        [
            html.H3("ABTC — Attractor-Based Trajectory Calculator"),
            html.P("Integrate a Lorenz-like attractor via ABTC rk4 helper."),
            html.Div(
                [
                    html.Label("Steps"),
                    dcc.Slider(id="abtc_steps", min=50, max=800, step=50, value=200),
                    html.Label("dt"),
                    dcc.Slider(
                        id="abtc_dt",
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
            html.Button("Run ABTC", id="abtc_run", n_clicks=0),
            dcc.Loading(dcc.Graph(id="abtc_traj"), type="dot"),
            dcc.Loading(dcc.Graph(id="abtc_series"), type="dot"),
            html.Div(id="abtc_shared_note", style={"paddingTop": "6px", "color": "#4b5563"}),
        ]
    )


def page_datasets():
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


# Landing experience callbacks -------------------------------------------------
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
def reflect_task_spec(spec: dict | None):
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
def populate_preview_table(data: dict | None):
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
def update_column_options(data: dict | None):
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
def populate_dataset_preview(data: dict | None):
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


@app.callback(Output("page", "children"), Input("url", "pathname"))
def router(pathname: str):
    if pathname == "/sim":
        return page_sim()
    if pathname == "/sheaf":
        return page_sheaf()
    if pathname == "/attr":
        return page_attr()
    if pathname == "/fractalllm":
        return page_fractalllm()
    if pathname == "/cog":
        return page_cognition()
    if pathname == "/dcrc":
        return page_dcrc()
    if pathname == "/self-tune":
        return page_self_tune()
    if pathname == "/bcp":
        return page_bcp()
    if pathname == "/abtc":
        return page_abtc()
    if pathname == "/datasets":
        return page_datasets()
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
    Output("phase3d", "figure", allow_duplicate=True),
    Output("series", "figure", allow_duplicate=True),
    Output("lle_out", "children", allow_duplicate=True),
    Input("show_shared_traj", "n_clicks"),
    prevent_initial_call=True,
)
def show_shared(n_clicks: int):
    traj = _shared_traj()
    if traj is None or getattr(traj, "size", 0) == 0:
        return dash.no_update, dash.no_update, "No shared dataset available."
    traj = np.array(traj, dtype=float)
    time = np.arange(traj.shape[0])
    # compute LLE on X using API
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
                z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                mode="lines",
                line=dict(width=2),
            )
        ]
    )
    fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig_ts = go.Figure()
    labels = ["X", "Y", "Z"]
    for idx, label in enumerate(labels):
        if idx < traj.shape[1]:
            fig_ts.add_trace(go.Scatter(x=time, y=traj[:, idx], name=label, mode="lines"))
    fig_ts.update_layout(height=300, xaxis_title="t")
    return fig3, fig_ts, f"Shared dataset LLE (X): {lle_val:.3f}"


def _shared_traj():
    """Return shared trajectory if present."""
    return getattr(server, "last_traj", None)


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
    rhos = _parse_numeric_list(rho_text, [])
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


@app.callback(
    Output("op_stats", "children"),
    Output("v2_map", "figure"),
    Input("build_op", "n_clicks"),
    prevent_initial_call=True,
)
def build_operator(n_clicks: int):
    traj = getattr(server, "last_traj", None)
    if traj is None:
        return "Run a simulation first or send a dataset from the Datasets page.", go.Figure()

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


@app.callback(
    Output("fh-summary", "children"),
    Output("fh-dq-graph", "figure"),
    Input("fh-run", "n_clicks"),
    Input("fh-run-cached", "n_clicks"),
    State("fh-coding-spec", "value"),
    State("fh-radius-bins", "value"),
    State("fh-radius-labels", "value"),
    prevent_initial_call=True,
)
def run_fractalhedron(
    run_clicks: int,
    cached_clicks: int,
    coding_spec: str,
    radius_bins: str | None,
    radius_labels: str | None,
):
    traj = _shared_traj()
    if traj is None:
        return "Run a simulation or load a dataset first.", go.Figure()

    traj_arr = np.asarray(traj, dtype=float)
    if traj_arr.ndim != 2 or traj_arr.shape[1] < 3:
        return "Shared trajectory must have at least X,Y,Z columns.", go.Figure()

    coding_params = None
    if coding_spec == "radius_bins":
        params: dict = {}
        if radius_bins:
            bins_list = _parse_numeric_list(radius_bins, [])
            if len(bins_list) < 2:
                return "Provide at least two radius bin edges.", go.Figure()
            params["bins"] = bins_list
        if radius_labels:
            if "bins" not in params:
                return "Specify radius bins before custom labels.", go.Figure()
            labels_list = _parse_label_list(radius_labels, [])
            if len(labels_list) != len(params["bins"]) - 1:
                return "Number of labels must equal len(bins) - 1.", go.Figure()
            params["labels"] = labels_list
        if params:
            coding_params = params

    try:
        # call API when cached button used so server stores section hits
        if dash.callback_context.triggered_id == "fh-run-cached":
            payload = {
                "section_hits": [],
                "coding_spec": coding_spec,
                "coding_params": coding_params or {},
                "k": 2,
                "Q": [0.0, 2.0],
            }
            resp = requests.post(f"{API}/fractalhedron/run", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            fh2 = {
                "alphabet": data["alphabet"],
                "D_q": {float(k): v for k, v in data["D_q"].items()},
                "T_q": {float(k): v for k, v in data["T_q"].items()},
                "ell_k": data["ell_k"],
                "constraints": data["constraints"],
            }
            faces = data["faces"]
        else:
            sym_seq = build_symbolic_sequence(traj_arr, coding_spec=coding_spec, coding_params=coding_params)
            fh2 = build_fractalhedron_k(sym_seq, k=2, Q=(0.0, 2.0))
            faces = fractal_face_flags(fh2, eps_p=1e-5, eps_D=1e-2)
    except Exception as exc:  # noqa: BLE001
        return f"Could not build FractalHedron: {exc}", go.Figure()

    qs = sorted(fh2["D_q"].keys())
    fig = go.Figure()
    fig.add_bar(x=[str(q) for q in qs], y=[fh2["D_q"][q] for q in qs], name="D_q")
    fig.update_layout(
        height=320,
        title="Fractal spectrum",
        xaxis_title="q",
        yaxis_title="D_q",
        yaxis=dict(range=[0, 1.2]),
    )

    zero_words = faces["symbolic_zero_words"][:4]
    summary = html.Div(
        [
            html.Div(f"Alphabet: {', '.join(fh2['alphabet']) or 'n/a'}"),
            html.Div(f"ell_2: {fh2['ell_k']:.4f}"),
            html.Div(f"Monotone: {fh2['constraints']['monotone_ok']} · Bounds: {fh2['constraints']['bounds_ok']}"),
            html.Div(f"Zero-prob words (sample): {zero_words or '—'}"),
            html.Div(f"Monofractal pairs: {faces['monofractal_pairs'] or '—'}"),
            html.Div(f"Near max dims: {faces['near_max_dim'] or '—'} · Near zero dims: {faces['near_zero_dim'] or '—'}"),
        ],
        style={"fontFamily": "JetBrains Mono, monospace", "lineHeight": "1.6"},
    )
    return summary, fig


@app.callback(
    Output("self-tune-session-id", "children"),
    Output("self-tune-timer", "disabled"),
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
    Output("self-tune-lambda", "figure"),
    Output("self-tune-sr", "figure"),
    Output("self-tune-attractor", "figure"),
    Output("self-tune-summary", "children"),
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
    Output("self-tune-lambda", "figure"),
    Output("self-tune-sr", "figure"),
    Output("self-tune-attractor", "figure"),
    Output("self-tune-summary", "children"),
    Input("self-tune-run", "n_clicks"),
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


@app.callback(
    Output("fllm_traj", "figure"),
    Output("fllm_switch", "figure"),
    Output("fllm_stats", "children"),
    Output("fllm_shared_note", "children"),
    Input("fllm_run", "n_clicks"),
    State("fllm_res_size", "value"),
    State("fllm_coupling", "value"),
    State("fllm_text", "value"),
    prevent_initial_call=True,
)
def run_fractalllm(n_clicks: int, res_size: int, coupling: float, text: str):
    shared = _shared_traj()
    if shared is not None and getattr(shared, "size", 0):
        traj = np.array(shared, dtype=float)
        sw = np.zeros_like(traj)
        fig_main = go.Figure(
            data=[
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2] if traj.shape[1] > 2 else np.zeros(traj.shape[0]),
                    mode="lines",
                    line=dict(width=2),
                    name="Shared dataset",
                )
            ]
        )
        fig_main.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
        fig_sw = go.Figure()
        note = "Using shared dataset trajectory."
        return fig_main, fig_sw, "Shared dataset visualized.", note

    payload = {"text": text or "", "reservoir_size": res_size, "coupling": coupling}
    resp = requests.post(f"{API}/fractalllm/process", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    traj = np.array(data.get("trajectory", []), dtype=float)
    sw = np.array(data.get("switch", []), dtype=float)

    fig_main = go.Figure()
    if traj.size:
        fig_main.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines",
                line=dict(width=2),
                name="Reservoir",
            )
        )
    fig_main.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig_sw = go.Figure()
    if sw.size:
        fig_sw.add_trace(
            go.Scatter3d(
                x=sw[:, 0],
                y=sw[:, 1],
                z=sw[:, 2],
                mode="lines",
                line=dict(width=2, color="firebrick"),
                name="Switch",
            )
        )
    fig_sw.update_layout(height=320, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    note = ""
    return fig_main, fig_sw, f"Index={data.get('index', 0.0):.3f}, points={traj.shape[0]}", note


def _parse_upload(contents: str) -> np.ndarray:
    """Decode base64 CSV upload into a 1D series."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    series = df.iloc[:, 0].astype(float).to_numpy()
    return series


def _parse_numeric_list(text: str | None, fallback: list[float]) -> list[float]:
    if text is None:
        return fallback
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    values: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return fallback
    return values or fallback


def _parse_label_list(text: str | None, fallback: list[str]) -> list[str]:
    if text is None:
        return fallback
    values = [p.strip() for p in text.split(",") if p.strip()]
    return values or fallback


def _parse_patterns_text(text: str | None, fallback: list[list[float]]) -> list[list[float]]:
    if text is None:
        return fallback
    patterns: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pattern = [float(tok) for tok in line.replace(" ", "").split(",") if tok]
        except ValueError:
            return fallback
        if pattern:
            patterns.append(pattern)
    return patterns or fallback


def _coerce_int(value: float | int | None, default: int, minimum: int = 0) -> int:
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = default
    return max(minimum, val)


def _coerce_float(value: float | int | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.callback(
    Output("lle_file_out", "children"),
    Input("lle_from_file_btn", "n_clicks"),
    State("upload-data", "contents"),
    prevent_initial_call=True,
)
def compute_lle_from_file(n_clicks: int, contents: str | None):
    if not contents:
        return "Upload a CSV first."
    try:
        series = _parse_upload(contents)
    except Exception as exc:  # noqa: BLE001
        return f"Could not read file: {exc}"
    resp = requests.post(
        f"{API}/metrics/lle",
        json={"series": series.tolist()},
        timeout=60,
    )
    resp.raise_for_status()
    val = float(resp.json()["lle"])
    return f"Uploaded series LLE: {val:.3f}"


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
    # overlay shared dataset if present
    shared = _shared_traj()
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
    alphabet = _parse_numeric_list(alphabet_text, DEFAULT_COG_ALPHABET)
    payload = {
        "max_lag": int(max_lag or 50),
        "amp": _coerce_float(amp, 2.0),
        "base_value": _coerce_float(base_value, 28.0),
        "dt": _coerce_float(dt_val, 0.01),
        "n_trials": _coerce_int(n_trials, 4, minimum=1),
        "n_steps": _coerce_int(n_steps, 5000, minimum=200),
        "burn_in": _coerce_int(burn_in, 500, minimum=0),
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
    patterns = _parse_patterns_text(pattern_text, DEFAULT_COG_PATTERNS)
    payload = {
        "patterns": [{"pattern": pattern} for pattern in patterns],
        "n_trials_per_concept": _coerce_int(trials_per, 100, minimum=1),
        "burn_in": _coerce_int(burn_in, 500, minimum=0),
        "amp": _coerce_float(amp, 2.0),
        "base_value": _coerce_float(base_value, 28.0),
        "dt": _coerce_float(dt_val, 0.01),
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


@app.callback(
    Output("bcp_plot", "figure"),
    Output("bcp_stats", "children"),
    Output("bcp_shared_note", "children"),
    Input("bcp_run", "n_clicks"),
    prevent_initial_call=True,
)
def run_bcp_section(n_clicks: int):
    shared = _shared_traj()
    if shared is not None and shared.shape[1] >= 2:
        pts = shared[:, [0, 2]] if shared.shape[1] > 2 else shared[:, :2]
        note = "Sectioned shared dataset."
    else:
        t = np.linspace(0, 2 * np.pi, 300)
        pts = np.stack([np.cos(t), np.sin(2 * t)], axis=1)
        note = "Sectioned sample curve."
    resp = requests.post(f"{API}/bcp/section", json={"points": pts.tolist()}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    sec = np.array(data.get("section", []), dtype=float)
    fig = go.Figure()
    if sec.size:
        fig.add_trace(go.Scatter(x=sec[:, 0], y=sec[:, 1], mode="markers", marker=dict(size=5)))
    fig.update_layout(height=360, xaxis_title="X", yaxis_title="Z")
    return fig, f"Section points: {data.get('count', 0)}", note


@app.callback(
    Output("abtc_traj", "figure"),
    Output("abtc_series", "figure"),
    Output("abtc_shared_note", "children"),
    Input("abtc_run", "n_clicks"),
    State("abtc_steps", "value"),
    State("abtc_dt", "value"),
    prevent_initial_call=True,
)
def run_abtc(n_clicks: int, steps: int, dt: float):
    shared = _shared_traj()
    if shared is not None and getattr(shared, "size", 0):
        traj = np.array(shared, dtype=float)
        from_shared = True
    else:
        resp = requests.post(f"{API}/abtc/integrate", json={"steps": steps, "dt": dt}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        traj = np.array(data.get("trajectory", []), dtype=float)
        from_shared = False

    fig3 = go.Figure()
    if traj.size:
        fig3.add_trace(
            go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="lines", line=dict(width=2))
        )
    fig3.update_layout(height=420, scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig_ts = go.Figure()
    labels = ["X", "Y", "Z"]
    t = np.arange(traj.shape[0]) * dt
    for idx, label in enumerate(labels):
        fig_ts.add_trace(go.Scatter(x=t, y=traj[:, idx], name=label, mode="lines"))
    fig_ts.update_layout(height=300, xaxis_title="t")
    note = "Using shared dataset trajectory." if from_shared else "Using ABTC-generated trajectory."
    return fig3, fig_ts, note


def _make_table(rows: list[dict]) -> html.Table:
    if not rows:
        return html.Table()
    cols = list(rows[0].keys())
    header = html.Tr([html.Th(c) for c in cols])
    body = [html.Tr([html.Td(r.get(c, "")) for c in cols]) for r in rows[:10]]
    return html.Table([header] + body, style={"borderCollapse": "collapse", "width": "100%", "border": "1px solid #e2e8f0"})


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
    Output("dataset_meta", "children"),
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
        # build 3D trajectory store if possible
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
    server.last_traj = np.array(traj, dtype=float)
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


def main() -> None:
    host = os.getenv("SACP_UI_HOST", "127.0.0.1")
    port = int(os.getenv("SACP_UI_PORT", "8050"))
    # Keep using run_server so Dash 2.x installs can launch the app.
    app.run_server(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
