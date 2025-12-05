from __future__ import annotations

from dash import Input, Output, State, dcc, html, no_update

from sacp_suite.ui.pages import register_page
from sacp_suite.ui.pages.common import API_BASE


MODULES = {
    "adr_krebs": {
        "title": "ADR–Krebs",
        "summary": "4-site metabolic tile (TCA, NADH, ΔΨ, ATP).",
        "links": [
            ("API: /api/v1/chemistry/simulate", f"{API_BASE}/api/v1/chemistry/simulate"),
            ("Notebook/Plugin: sacp_suite.modules.chemistry.adr_krebs_plugin", None),
        ],
    },
    "adr_hemostasis": {
        "title": "ADR–Hemostasis",
        "summary": "4-site vascular tile (prostanoid, thrombus, LDL/plaque, cytokine).",
        "links": [
            ("BaseDynamics plugin 'adr_hemostasis' (sacp_suite.modules.sacp_x.adr_hemostasis)", None),
        ],
    },
    "adr_bioelectric": {
        "title": "ADR Bioelectric",
        "summary": "Multisite wound/repair demo (ΔVmem per tile).",
        "links": [
            ("API: /api/v1/chemistry/simulate (mode=wound)", f"{API_BASE}/api/v1/chemistry/simulate"),
        ],
    },
    "bouquet": {
        "title": "Bouquet stack",
        "summary": "ADR stack with bouquet speed bound; use run/scan for CLC checks.",
        "links": [
            ("API: /api/v1/bouquet/run", f"{API_BASE}/api/v1/bouquet/run"),
            ("API: /api/v1/bouquet/scan", f"{API_BASE}/api/v1/bouquet/scan"),
        ],
    },
}


def layout():
    options = [{"label": info["title"], "value": key} for key, info in MODULES.items()]
    return html.Div(
        [
            html.H3("ADR models", className="section-title"),
            html.P(
                "Entry point for Autocatalytic Duffing Ring (ADR) surrogates. Select a module to view its info and entry points."
            ),
            html.Div(
                [
                    html.Label("Choose ADR module"),
                    dcc.Dropdown(
                        id="adr-module-select",
                        options=options,
                        value=options[0]["value"] if options else None,
                        clearable=False,
                    ),
                ],
                className="card",
            ),
            html.Div(id="adr-module-card", className="card"),
            html.Div(
                "Tip: in notebooks or ABTC registry, instantiate ADR models by key: 'adr_krebs', "
                "'adr_hemostasis', or call bouquet.run/scan via the API."
            ),
        ]
    )


def register_callbacks(app):
    @app.callback(Output("adr-module-card", "children"), Input("adr-module-select", "value"))
    def _render_card(module_key: str):
        if not module_key or module_key not in MODULES:
            return no_update
        info = MODULES[module_key]
        links = []
        for text, href in info["links"]:
            if href:
                links.append(html.A(text, href=href, target="_blank"))
            else:
                links.append(html.Span(text))
            links.append(html.Br())
        return html.Div(
            [
                html.H4(info["title"]),
                html.P(info["summary"]),
                html.Div(links[:-1] if links else []),
            ]
        )


register_page(
    "adr",
    name="ADR",
    path="/adr",
    layout=layout,
    register_callbacks=register_callbacks,
    order=85,
)
