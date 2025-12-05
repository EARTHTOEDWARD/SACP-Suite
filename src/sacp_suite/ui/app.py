"""Dash shell for SACP Suite using pluggable page modules."""

from __future__ import annotations

import os

from dash import Dash, Input, Output, dcc, html

from sacp_suite.ui.pages import bootstrap_pages, find_page_by_path, get_pages, wire_callbacks
from sacp_suite.ui.pages.explainers import explainer_component


def _create_app() -> Dash:
    external_stylesheets = [
        # Add external stylesheets if needed; assets/ is served automatically.
    ]
    app = Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
        title="SACP Suite",
    )
    return app


# Create app + server first so helper modules can reference `server`
app = _create_app()
server = app.server

# Import & register all pages
bootstrap_pages()
wire_callbacks(app)


def _navbar():
    all_pages = get_pages()
    pages = [p for p in all_pages if p.id != "datasets"]
    dataset_page = next((p for p in all_pages if p.id == "datasets"), None)
    links = []
    for page in pages:
        links.append(
            dcc.Link(
                page.name,
                href=page.path,
                className="nav-link",
            )
        )
    if dataset_page is not None:
        links.append(
            dcc.Link(
                dataset_page.name,
                href=dataset_page.path,
                className="nav-link nav-secondary",
                title="Datasets workspace",
                style={"marginLeft": "auto"},
            )
        )
    return html.Div(
        links,
        className="sacp-navbar",
    )


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        # Shared stores used across pages
        dcc.Store(id="shared_traj"),
        dcc.Store(id="task_spec_store"),
        dcc.Store(id="landing_stage", data="prompt"),
        dcc.Store(id="ingestion_preview_store"),
        dcc.Store(id="dataset_builder_store"),
        html.Header(
            [
                html.H1("Strange Attractor Control Panel Suite", className="sacp-title"),
                _navbar(),
            ],
            className="sacp-header",
        ),
        html.Div(id="page-content", className="sacp-page-content"),
    ],
    className="sacp-root",
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def _render_page(pathname: str):
    page = find_page_by_path(pathname or "/")
    if page is None:
        return html.Div(
            [
                html.H2("404 – Page not found"),
                html.P(f"No page is registered for path {pathname!r}."),
            ],
            style={"padding": "2rem"},
        )

    try:
        content = page.layout()
        explainer = explainer_component(page.id)
        if explainer is not None:
            return html.Div([explainer, content], className="sacp-page-shell")
        return content
    except Exception as exc:  # noqa: BLE001
        from sacp_suite.ui.pages import _error_layout  # type: ignore[attr-defined]

        return _error_layout(page.id, f"Error rendering page layout: {exc}")


def main() -> None:
    """Entry point for `sacp-suite` console script."""
    host = os.getenv("SACP_HOST", "127.0.0.1")
    port = int(os.getenv("SACP_PORT", "8050"))
    debug = os.getenv("SACP_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
