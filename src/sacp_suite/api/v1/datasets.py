"""Tenant-aware dataset endpoints (stubs wrapping future storage service)."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sacp_suite.api.deps import get_current_user, get_db_session
from sacp_suite.db import models
from sacp_suite.services import datasets as dataset_service

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


class DatasetCreate(BaseModel):
    workspace_id: str
    name: str
    description: str | None = None
    content_b64: str


class DatasetPreviewOut(BaseModel):
    columns: list[str]
    rows: list[list[Any]]


class DatasetOut(BaseModel):
    id: str
    workspace_id: str
    name: str
    description: str | None
    n_rows: int | None
    n_cols: int | None

    class Config:
        orm_mode = True


@router.post("/", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> models.Dataset:
    try:
        raw_bytes = base64.b64decode(payload.content_b64)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 content.")

    dataset, _meta = dataset_service.save_dataset_file(
        db=db,
        workspace_id=payload.workspace_id,
        user_id=None,
        name=payload.name,
        description=payload.description,
        file_bytes=raw_bytes,
    )
    return dataset


@router.get("/", response_model=list[DatasetOut])
def list_datasets(
    workspace_id: str,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> list[models.Dataset]:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Dataset listing not implemented yet."
    )


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(
    dataset_id: str,
    limit: int = 200,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> DatasetPreviewOut:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Dataset preview not implemented yet."
    )
