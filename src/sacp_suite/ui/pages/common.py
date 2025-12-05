from __future__ import annotations

import base64
import io
import os
from typing import List

import numpy as np
import pandas as pd

API_BASE = f"http://{os.getenv('SACP_BIND', '127.0.0.1')}:{os.getenv('SACP_PORT', '8000')}"
COLAB_NOTEBOOK_URL = os.getenv(
    "SACP_COLAB_NOTEBOOK_URL",
    "https://colab.research.google.com/github/your-org/your-repo/blob/main/notebooks/sacp_suite_colab.ipynb",
)

# Simple in-process shared trajectory cache (replaces server attribute)
_shared_traj = None


def get_shared_traj():
    return _shared_traj


def set_shared_traj(traj) -> None:
    global _shared_traj
    _shared_traj = traj


def parse_upload(contents: str) -> np.ndarray:
    """Decode base64 CSV upload into a 1D series."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    series = df.iloc[:, 0].astype(float).to_numpy()
    return series


def parse_numeric_list(text: str | None, fallback: List[float]) -> List[float]:
    if text is None:
        return fallback
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    values: List[float] = []
    for part in parts:
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return fallback
    return values or fallback


def parse_label_list(text: str | None, fallback: List[str]) -> List[str]:
    if text is None:
        return fallback
    values = [p.strip() for p in text.split(",") if p.strip()]
    return values or fallback


def parse_patterns_text(text: str | None, fallback: List[List[float]]) -> List[List[float]]:
    if text is None:
        return fallback
    patterns: List[List[float]] = []
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


def coerce_int(value: float | int | None, default: int, minimum: int = 0) -> int:
    try:
        val = int(value)
    except (TypeError, ValueError):
        val = default
    return max(minimum, val)


def coerce_float(value: float | int | None, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def colab_launch_url(token: str | None = None) -> str:
    """
    Build a Colab URL with the API base prefilled. Token is optional and should
    be provided by the user at runtime rather than hardcoded.
    """
    base_param = f"apiBase={API_BASE}"
    token_param = f"&token={token}" if token else ""
    sep = "&" if "?" in COLAB_NOTEBOOK_URL else "?"
    return f"{COLAB_NOTEBOOK_URL}{sep}{base_param}{token_param}"


# Dark theme configuration for Plotly graphs
PLOTLY_DARK_LAYOUT = {
    'paper_bgcolor': '#0a0a0a',
    'plot_bgcolor': '#000000',
    'font': {
        'color': '#F5F5F5',
        'family': 'Inter, -apple-system, sans-serif'
    },
    'xaxis': {
        'gridcolor': '#222222',
        'zerolinecolor': '#333333',
        'color': '#B0B0B0'
    },
    'yaxis': {
        'gridcolor': '#222222',
        'zerolinecolor': '#333333',
        'color': '#B0B0B0'
    },
    'scene': {  # For 3D plots
        'xaxis': {'gridcolor': '#222222', 'color': '#B0B0B0'},
        'yaxis': {'gridcolor': '#222222', 'color': '#B0B0B0'},
        'zaxis': {'gridcolor': '#222222', 'color': '#B0B0B0'},
        'bgcolor': '#000000'
    },
    'colorway': [
        '#FFFFFF',  # White - primary trace
        '#00CED1',  # Cyan
        '#4A9EFF',  # Bright blue
        '#1DE9B6',  # Bright cyan/green
        '#7B68EE',  # Medium slate blue
        '#00BFFF',  # Deep sky blue
        '#48D1CC',  # Medium turquoise
        '#87CEEB',  # Sky blue
    ]
}


def dark_table_style():
    """Return consistent dark styling for DataTables."""
    return {
        'style_table': {
            'backgroundColor': '#0a0a0a',
            'maxHeight': '400px',
            'overflowY': 'auto'
        },
        'style_header': {
            'backgroundColor': '#141414',
            'color': '#F5F5F5',
            'fontWeight': '600',
            'borderBottom': '2px solid #008B8B'
        },
        'style_cell': {
            'backgroundColor': '#0a0a0a',
            'color': '#F5F5F5',
            'fontFamily': 'JetBrains Mono, monospace',
            'border': '1px solid #222222',
            'textAlign': 'left'
        },
        'style_data_conditional': [
            {
                'if': {'state': 'active'},
                'backgroundColor': '#1a1a1a',
                'border': '1px solid #00CED1'
            }
        ]
    }
