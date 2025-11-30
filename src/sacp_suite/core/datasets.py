"""Dataset registry and loader for bundled example datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repo root (/.../SACP SUITE)
MANIFEST_PATH = ROOT / "datasets" / "manifest.json"

# Fallback: tolerate editable installs where the relative depth might differ
if not MANIFEST_PATH.exists():
    alt_root = Path(__file__).resolve().parents[2]
    alt_manifest = alt_root / "datasets" / "manifest.json"
    if alt_manifest.exists():
        ROOT = alt_root
        MANIFEST_PATH = alt_manifest


@dataclass
class DatasetInfo:
    id: str
    name: str
    description: str
    path: Path
    columns: List[str]
    count_hint: int | None = None
    source: str | None = None


def _load_manifest() -> list[DatasetInfo]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open() as f:
        raw = json.load(f)
    out = []
    for item in raw:
        out.append(
            DatasetInfo(
                id=item["id"],
                name=item.get("name", item["id"]),
                description=item.get("description", ""),
                path=ROOT / item["path"],
                columns=item.get("columns", []),
                count_hint=item.get("count_hint"),
                source=item.get("source"),
            )
        )
    return out


def list_datasets() -> list[DatasetInfo]:
    return _load_manifest() + _load_uploads()


def preview_dataset(dataset_id: str, limit: int = 1000) -> dict[str, Any]:
    for ds in list_datasets():
        if ds.id == dataset_id:
            if not ds.path.exists():
                raise FileNotFoundError(f"Dataset file missing: {ds.path}")
            df = pd.read_csv(ds.path, nrows=limit)
            return {
                "id": ds.id,
                "name": ds.name,
                "description": ds.description,
                "columns": df.columns.tolist(),
                "rows": df.to_dict(orient="records"),
                "total_estimate": ds.count_hint,
                "source": ds.source,
            }
    raise KeyError(f"Unknown dataset '{dataset_id}'")


# Upload handling -------------------------------------------------------------
UPLOADS_DIR = ROOT / "var" / "uploads" / "datasets"
UPLOADS_MANIFEST = UPLOADS_DIR / "uploads.json"


def _load_uploads() -> list[DatasetInfo]:
    if not UPLOADS_MANIFEST.exists():
        return []
    with UPLOADS_MANIFEST.open() as f:
        raw = json.load(f)
    out = []
    for item in raw:
        out.append(
            DatasetInfo(
                id=item["id"],
                name=item.get("name", item["id"]),
                description=item.get("description", ""),
                path=ROOT / item["path"],
                columns=item.get("columns", []),
                count_hint=item.get("count_hint"),
                source=item.get("source", "upload"),
            )
        )
    return out


def _persist_uploads(entries: list[dict]) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with UPLOADS_MANIFEST.open("w") as f:
        json.dump(entries, f, indent=2)


def register_upload(name: str, description: str, columns: list[str], content: bytes) -> DatasetInfo:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "upload"
    file_path = UPLOADS_DIR / f"{safe}.csv"
    file_path.write_bytes(content)
    existing = []
    if UPLOADS_MANIFEST.exists():
        with UPLOADS_MANIFEST.open() as f:
            existing = json.load(f)
    entry = {
        "id": safe,
        "name": name,
        "description": description,
        "path": str(file_path.relative_to(ROOT)),
        "columns": columns,
        "count_hint": None,
        "source": "upload",
    }
    existing = [e for e in existing if e.get("id") != safe]
    existing.append(entry)
    _persist_uploads(existing)
    return DatasetInfo(
        id=entry["id"],
        name=name,
        description=description,
        path=file_path,
        columns=columns,
        count_hint=None,
        source="upload",
    )
