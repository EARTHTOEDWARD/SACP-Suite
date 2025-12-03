"""
Page registry for the SACP Suite Dash UI.

Each page module under `sacp_suite.ui.pages` should call `register_page(...)`
at import time. The Dash shell (`ui/app.py`) then:

    1) calls `bootstrap_pages()` to import all pages defensively
    2) calls `wire_callbacks(app)` so each page can attach its callbacks

If a page import or callback registration fails, we register a fallback
"error layout" instead of crashing the whole app.
"""

from __future__ import annotations

import importlib
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dash import html

LayoutFn = Callable[[], Any]
CallbacksFn = Callable[[Any], None]


@dataclass
class Page:
    id: str
    name: str
    path: str
    layout: LayoutFn
    register_callbacks: Optional[CallbacksFn] = None
    icon: Optional[str] = None
    order: int = 0
    error: Optional[str] = None


_PAGES: Dict[str, Page] = {}


def register_page(
    page_id: str,
    *,
    name: str,
    path: str,
    layout: LayoutFn,
    register_callbacks: Optional[CallbacksFn] = None,
    icon: Optional[str] = None,
    order: int = 0,
) -> None:
    """
    Called by page modules during import to register themselves.

    Example (inside ui/pages/simulator.py):

        def layout():
            ...

        def register_callbacks(app):
            @app.callback(...)
            def _cb(...):
                ...

        register_page(
            "simulator",
            name="Simulator",
            path="/simulator",
            layout=layout,
            register_callbacks=register_callbacks,
            order=0,
        )
    """
    _PAGES[page_id] = Page(
        id=page_id,
        name=name,
        path=path,
        layout=layout,
        register_callbacks=register_callbacks,
        icon=icon,
        order=order,
        error=None,
    )


def _error_layout(page_id: str, error_text: str):
    return html.Div(
        [
            html.H2(f"{page_id} page failed to load"),
            html.P(
                "This page encountered an error during import or callback "
                "registration. Other pages are still available."
            ),
            html.Pre(error_text[:4000]),
        ],
        style={"padding": "2rem", "color": "#b00020"},
    )


def _register_error_page(page_id: str, exc: BaseException, order: int = 999) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _PAGES[page_id] = Page(
        id=page_id,
        name=f"{page_id.title()} (error)",
        path=f"/{page_id}",
        layout=lambda pid=page_id, msg=tb: _error_layout(pid, msg),
        register_callbacks=None,
        icon=None,
        order=order,
        error=tb,
    )


def bootstrap_pages() -> None:
    """
    Import all known page modules with defensive error handling.

    Honours optional env var SACP_ENABLED_PAGES, e.g.
        SACP_ENABLED_PAGES="simulator,datasets,chemistry"
    """
    module_names = [
        "home",
        "simulator",
        "sheaf",
        "attractorhedron",
        "fractal_llm",
        "cognition",
        "dcrc",
        "self_tuning",
        "bcp",
        "abtc",
        "datasets",
        "chemistry",
        "frac_chem_sprott",
    ]

    enabled_env = os.getenv("SACP_ENABLED_PAGES")
    enabled_ids = None
    if enabled_env:
        enabled_ids = {s.strip() for s in enabled_env.split(",") if s.strip()}

    for order, module_name in enumerate(module_names):
        page_id = module_name

        if enabled_ids is not None and page_id not in enabled_ids:
            continue

        if page_id in _PAGES:
            # Already imported / registered.
            continue

        module_path = f"sacp_suite.ui.pages.{module_name}"
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            # Do not crash the whole application; register an error placeholder.
            _register_error_page(page_id, exc, order=order)


def get_pages() -> List[Page]:
    """Return all registered pages, sorted by `order` then name."""
    return sorted(_PAGES.values(), key=lambda p: (p.order, p.name.lower()))


def find_page_by_path(pathname: str) -> Optional[Page]:
    """Resolve a URL path (e.g. '/simulator') to a Page object."""
    if not pathname:
        return None

    if pathname != "/" and pathname.endswith("/"):
        pathname = pathname[:-1]

    for page in _PAGES.values():
        if page.path == pathname:
            return page

    if pathname == "/":
        pages = get_pages()
        return pages[0] if pages else None

    return None


def wire_callbacks(app: Any) -> None:
    """
    Ask each page to register its callbacks on the given Dash app.

    Any failure is converted into an error layout for that page only.
    """
    for page in get_pages():
        if page.register_callbacks is None:
            continue
        try:
            page.register_callbacks(app)
        except Exception as exc:  # noqa: BLE001
            _register_error_page(page.id, exc, order=page.order)
