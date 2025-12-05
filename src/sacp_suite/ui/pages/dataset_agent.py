from __future__ import annotations

from typing import Dict, List, Any

import dash
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html, ALL
from dash.exceptions import PreventUpdate

from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE, set_shared_traj, PLOTLY_DARK_LAYOUT, colab_launch_url

API = API_BASE

DOMAIN_HINTS = {
    "lorenz": "research",
    "rossler": "research",
    "hr_": "clinical",
    "hindmarsh": "clinical",
}


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Dataset Agent (placeholder)", className="section-title"),
                    html.P(
                        "Draft plans to locate and evaluate datasets for strange-attractor or dynamical-systems work. "
                        "This is a scaffold for the full agent described in the blueprint.",
                        className="status-text",
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("What do you need a dataset for?"),
                            dcc.Textarea(
                                id="da_intent",
                                placeholder="Example: EEG chaos indicators for seizure onset; or HVAC sensor logs to detect bifurcations.",
                                style={"width": "100%", "minHeight": "120px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Domain"),
                            dcc.Dropdown(
                                id="da_domain",
                                options=[
                                    {"label": "Finance / risk", "value": "finance"},
                                    {"label": "Chemistry / bioelectric", "value": "chemistry"},
                                    {"label": "Control / engineering", "value": "control"},
                                    {"label": "Clinical / neuro", "value": "clinical"},
                                    {"label": "Other / research", "value": "other"},
                                ],
                                value="finance",
                                clearable=False,
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Data modality"),
                            dcc.Dropdown(
                                id="da_modality",
                                options=[
                                    {"label": "Time series", "value": "time_series"},
                                    {"label": "Images", "value": "images"},
                                    {"label": "Graphs / networks", "value": "graphs"},
                                    {"label": "Text / logs", "value": "text"},
                                    {"label": "Mixed", "value": "mixed"},
                                ],
                                value="time_series",
                                clearable=False,
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Constraints"),
                            dcc.Checklist(
                                id="da_constraints",
                                options=[
                                    {"label": "Open-license only", "value": "open"},
                                    {"label": "PII/PHI safe", "value": "pii_safe"},
                                    {"label": "Synthetic acceptable", "value": "synthetic_ok"},
                                    {"label": "Need citations / provenance", "value": "provenance"},
                                ],
                                value=["open", "provenance"],
                                className="inline-check",
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Button("Draft dataset plan", id="da_run", n_clicks=0, className="primary-btn", style={"marginTop": "12px"}),
                    html.Div(id="da_status", className="status-text", style={"marginTop": "6px"}),
                    html.Button("Find candidate datasets", id="da_find", n_clicks=0, className="primary-btn", style={"marginTop": "12px"}),
                    dcc.Store(id="da_candidates"),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.H4("Plan preview", className="section-title"),
                    html.Div(id="da_plan", className="task-summary-box"),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
            html.Div(
                [
                    html.H4("What this agent will eventually do", className="section-title"),
                    html.Ul(
                        [
                            html.Li("Parse your intent, domain, and modality; map to relevant corpora (papers, repos, registries)."),
                            html.Li("Search internal/curated datasets and external registries with licensing filters."),
                            html.Li("Evaluate candidates with quick health checks (coverage, noise, stationarity, chaos indicators)."),
                            html.Li("Produce provenance-aware briefs with suggested preprocessing and verification steps."),
                            html.Li("Integrate with the Suite: push datasets to Simulator/Attractorhedron/Fractal LLM pipelines."),
                        ]
                    ),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
            html.Div(
                [
                    html.H4("Candidate datasets (internal + external stubs)", className="section-title"),
                    html.Div(id="da_candidates_view", className="card"),
                    html.Div(id="da_candidate_status", className="status-text", style={"marginTop": "6px"}),
                ],
                className="card",
                style={"marginTop": "12px"},
            ),
                    html.Div(
                        [
                            html.H4("Preview + quick health", className="section-title"),
                            dcc.Loading(dcc.Graph(id="da_preview_fig"), type="dot"),
                            html.Div(id="da_preview_meta", className="status-text", style={"marginTop": "6px"}),
                            html.Div(id="da_remote_hint", className="status-text", style={"marginTop": "6px"}),
                        ],
                        className="card",
                        style={"marginTop": "12px"},
                    ),
        ]
    )


def _draft_plan(intent: str, domain: str, modality: str, constraints: List[str]) -> List[str]:
    intent = intent.strip() or "Find a dataset for chaos analysis."
    domain_map = {
        "finance": "volatility regimes, order-book depth, macro time series",
        "chemistry": "bioelectric potentials, metabolic tiles, reaction flows",
        "control": "sensor/actuator logs, HVAC/ICS time series, SCADA-like traces",
        "clinical": "EEG, ECG, gait sensors, ICU waveforms",
        "other": "research benchmarks and public attractor galleries",
    }
    modality_text = {
        "time_series": "multivariate time series with sufficient length for Lyapunov/embedding",
        "images": "image/volume data (needs feature extraction before chaos metrics)",
        "graphs": "network/graph signals (may need graph-aware embeddings)",
        "text": "logs/text that encode temporal events (will require parsing to series)",
        "mixed": "mixed modalities; plan will propose per-modality adapters",
    }
    constraint_notes = []
    if "open" in constraints:
        constraint_notes.append("license: permissive (CC-BY/CC0/MIT)")
    if "pii_safe" in constraints:
        constraint_notes.append("PII/PHI-safe sources only")
    if "provenance" in constraints:
        constraint_notes.append("capture provenance and citations")
    if "synthetic_ok" in constraints:
        constraint_notes.append("synthetic or generated acceptable")

    steps = [
        f"Intent: {intent}",
        f"Domain focus: {domain_map.get(domain, domain)}; modality: {modality_text.get(modality, modality)}.",
        "Search scope: curated Suite datasets (Strange_Attractor, synthetic builders), + public registries (e.g., UCI/PhysioNet/finance APIs) — placeholder only, no live search yet.",
        "Screening: quick stats (length, missingness, sampling), chaos cues (Lyapunov proxy, recurrence density), noise estimate; flag license/PII constraints.",
        "Output: ranked shortlist with provenance, suggested preprocessing, and push-ready trajectories for Simulator/Attractorhedron/Fractal LLM.",
    ]
    if constraint_notes:
        steps.append("Constraints: " + "; ".join(constraint_notes))
    return steps


def _score_dataset(ds: Dict[str, Any], constraints: List[str]) -> str:
    notes = []
    if ds.get("license"):
        notes.append(f"license: {ds['license']}")
    if ds.get("pii_safe") is False and "pii_safe" in constraints:
        notes.append("⚠ may contain PII/PHI")
    if ds.get("pii_safe") is True:
        notes.append("PII-safe")
    return "; ".join(notes)


def _domain_hint(ds_id: str, name: str) -> str:
    low = (ds_id + " " + name).lower()
    for key, val in DOMAIN_HINTS.items():
        if key in low:
            return val
    return "research"


def _fetch_internal_catalog() -> List[Dict[str, Any]]:
    resp = requests.get(f"{API}/datasets", timeout=10)
    resp.raise_for_status()
    items = resp.json().get("datasets", [])
    for item in items:
        item["kind"] = "internal"
        item["domain"] = item.get("domain") or _domain_hint(item.get("id", ""), item.get("name", ""))
        item["modality"] = item.get("modality") or "time_series"
    return items


def _fetch_external_stubs() -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{API}/datasets/registry", timeout=5)
        resp.raise_for_status()
        out = resp.json().get("external", [])
        for o in out:
            o["kind"] = "external"
    except Exception:
        out = []
    return out


def _filter_candidates(domain: str, modality: str, constraints: List[str]) -> List[Dict[str, Any]]:
    internal = _fetch_internal_catalog()
    external = _fetch_external_stubs()
    out: List[Dict[str, Any]] = []
    for ds in internal:
        if domain != "other" and ds.get("domain") not in {domain, "research"}:
            continue
        if ds.get("modality") != modality and modality != "mixed":
            continue
        out.append(ds)
    for stub in external:
        if domain != "other" and stub.get("domain") != domain:
            continue
        if stub.get("modality") != modality and modality != "mixed":
            continue
        out.append(stub)
    if "pii_safe" in constraints:
        out = [ds for ds in out if ds.get("pii_safe") or ds.get("kind") == "internal"]
    return out


def _build_candidate_card(ds: Dict[str, Any], constraints: List[str]) -> html.Div:
    if ds.get("kind") == "internal":
        button = html.Button(
            "Preview & send",
            id={"type": "da-preview-btn", "dsid": ds["id"]},
            n_clicks=0,
            className="primary-btn",
            style={"marginTop": "6px"},
        )
    else:
        button = html.Div("External stub (hook up registry API to enable).", className="status-text")
    return html.Div(
        [
            html.Div(ds.get("name") or ds.get("source", "Dataset"), style={"fontWeight": 700}),
            html.Div(ds.get("description", ""), className="status-text"),
            html.Div(_score_dataset(ds, constraints), className="status-text"),
            button,
        ],
        style={"padding": "10px", "borderBottom": "1px solid #e5e7eb"},
    )


def _quick_health(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    n = len(rows)
    cols = columns or (list(rows[0].keys()) if rows else [])
    chaos_hint = "OK" if n >= 500 and len(cols) >= 3 else "Weak (need >=500 rows and >=3 dims)"
    return f"rows={n}, dims={len(cols)}, chaos-readiness={chaos_hint}"


def _to_traj(rows: List[Dict[str, Any]], columns: List[str]) -> List[List[float]]:
    if not rows or not columns:
        return []
    cols = columns[:3]
    traj = []
    for r in rows:
        vec = [float(r.get(c, 0.0)) for c in cols]
        if len(vec) == 3:
            traj.append(vec)
    return traj


def register_callbacks(app):
    @app.callback(
        Output("da_plan", "children"),
        Output("da_status", "children"),
        Output("da_candidates", "data", allow_duplicate=True),
        Input("da_run", "n_clicks"),
        State("da_intent", "value"),
        State("da_domain", "value"),
        State("da_modality", "value"),
        State("da_constraints", "value"),
        prevent_initial_call=True,
    )
    def draft_dataset_plan(n_clicks: int, intent: str, domain: str, modality: str, constraints: List[str]):
        if not n_clicks:
            raise PreventUpdate
        steps = _draft_plan(intent or "", domain or "other", modality or "time_series", constraints or [])
        plan = html.Ul([html.Li(step) for step in steps], style={"lineHeight": "1.6"})
        status = "Drafted plan (placeholder, no live search yet)."
        # seed candidates too
        candidates = _filter_candidates(domain or "other", modality or "time_series", constraints or [])
        return plan, status, candidates

    @app.callback(
        Output("da_candidates", "data", allow_duplicate=True),
        Output("da_candidate_status", "children"),
        Input("da_find", "n_clicks"),
        State("da_domain", "value"),
        State("da_modality", "value"),
        State("da_constraints", "value"),
        prevent_initial_call=True,
    )
    def find_candidates(n_clicks: int, domain: str, modality: str, constraints: List[str]):
        if not n_clicks:
            raise PreventUpdate
        candidates = _filter_candidates(domain or "other", modality or "time_series", constraints or [])
        status = f"Found {len(candidates)} candidates (internal + placeholder external)."
        return candidates, status

    @app.callback(
        Output("da_candidates_view", "children"),
        Input("da_candidates", "data"),
        State("da_constraints", "value"),
    )
    def render_candidates(candidates: List[Dict[str, Any]] | None, constraints: List[str]):
        if not candidates:
            return html.Div("No candidates yet — click Find candidate datasets.", className="status-text")
        cards = [_build_candidate_card(ds, constraints or []) for ds in candidates]
        return html.Div(cards)

    @app.callback(
        Output("da_preview_fig", "figure", allow_duplicate=True),
        Output("da_preview_meta", "children", allow_duplicate=True),
        Output("da_candidate_status", "children", allow_duplicate=True),
        Output("da_remote_hint", "children", allow_duplicate=True),
        Output("shared_traj", "data", allow_duplicate=True),
        Input({"type": "da-preview-btn", "dsid": ALL}, "n_clicks"),
        State("da_candidates", "data"),
        prevent_initial_call=True,
    )
    def preview_dataset(n_clicks_list, candidates: List[Dict[str, Any]] | None):
        if not candidates:
            raise PreventUpdate
        triggered = dash.callback_context.triggered
        if not triggered:
            raise PreventUpdate
        trig_id = triggered[0]["prop_id"].split(".")[0]
        try:
            dsid = eval(trig_id).get("dsid")  # nosec - Dash pattern id is controlled
        except Exception:
            raise PreventUpdate
        target = next((c for c in candidates if c.get("id") == dsid), None)
        if not target or target.get("kind") != "internal":
            return (
                dash.no_update,
                "External stub: connect registry search to enable preview.",
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        try:
            resp = requests.post(f"{API}/datasets/preview", json={"dataset_id": dsid, "limit": 1500}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return dash.no_update, f"Preview failed: {exc}", dash.no_update, dash.no_update, dash.no_update

        rows = data.get("rows", [])
        columns = data.get("columns", [])
        health = _quick_health(rows, columns)
        traj = _to_traj(rows, columns)

        # Quick LLE on first column if present
        lle_txt = ""
        if rows and columns:
            try:
                series = [float(r.get(columns[1 if columns[0].lower() == "t" and len(columns) > 1 else 0], 0.0)) for r in rows[:1200]]
                resp = requests.post(f"{API}/metrics/lle", json={"series": series, "m": 6, "tau": 4}, timeout=10)
                resp.raise_for_status()
                lle_val = resp.json().get("lle")
                if lle_val is not None:
                    lle_txt = f" · LLE≈{float(lle_val):.3f}"
            except Exception:
                lle_txt = ""

        fig = go.Figure()
        if rows and len(columns) >= 2:
            x = [r.get(columns[0], 0) for r in rows]
            y = [r.get(columns[1], 0) for r in rows]
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=4)))
            fig.update_layout(**PLOTLY_DARK_LAYOUT)
            fig.update_layout(height=320, xaxis_title=columns[0], yaxis_title=columns[1])
        meta = f"{target.get('name', dsid)} — {target.get('license', '')}; {health}{lle_txt}"
        remote_hint = dash.no_update
        total_est = data.get("total_estimate") or len(rows)
        if isinstance(total_est, (int, float)) and total_est >= 200000:
            url = colab_launch_url()
            remote_hint = html.Div(
                [
                    html.Span("Large dataset detected. Consider using remote GPU via Colab: "),
                    html.A("Open in Colab", href=url, target="_blank", rel="noopener"),
                    html.Span(" (paste your API token in the notebook)."),
                ]
            )
        if traj:
            set_shared_traj(traj)
            status = f"Previewed {dsid}; shared trajectory ready (first 3 dims)."
        else:
            status = f"Previewed {dsid}; not enough dims to share."
        return fig, meta, status, remote_hint, traj or dash.no_update

register_page(
    "dataset_agent",
    name="Dataset Agent",
    path="/dataset-agent",
    layout=layout,
    register_callbacks=register_callbacks,
    order=8,
)
