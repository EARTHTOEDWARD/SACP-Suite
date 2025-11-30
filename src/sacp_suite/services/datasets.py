"""Dataset storage service (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sqlalchemy.orm import Session

from sacp_suite.db import models

DATASETS_ROOT = Path("var/uploads/datasets")


def save_dataset_file(
    db: Session,
    workspace_id: str,
    user_id: str | None,
    name: str,
    description: str | None,
    file_bytes: bytes,
) -> tuple[models.Dataset, dict]:
    """Persist a dataset file locally and return Dataset model + basic metadata.

    This is a placeholder; a future version should store files under per-tenant prefixes
    and write a DB row. Here we just write to disk and return an in-memory model.
    """

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    filename = f"{name}.csv"
    path = DATASETS_ROOT / filename
    path.write_bytes(file_bytes)

    # derive simple metadata
    df = pd.read_csv(path, nrows=1000)
    meta = {"columns": df.columns.tolist(), "preview_rows": len(df)}

    dataset = models.Dataset(
        id=None,  # type: ignore[arg-type]
        workspace_id=workspace_id,
        name=name,
        description=description,
        source_type=models.DatasetSourceType.UPLOAD,
        storage_uri=str(path),
        schema_json={"columns": df.columns.tolist()},
        n_rows=None,
        n_cols=len(df.columns),
        created_by_id=user_id,
    )
    return dataset, meta
