"""Workspace and organisation endpoints (stubs)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sacp_suite.api.deps import get_current_user, get_db_session
from sacp_suite.db import models

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


class WorkspaceCreate(BaseModel):
    organisation_id: str
    name: str
    description: str | None = None


class WorkspaceOut(BaseModel):
    id: str
    organisation_id: str
    name: str
    description: str | None

    class Config:
        orm_mode = True


@router.get("/", response_model=list[WorkspaceOut])
def list_workspaces(
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> list[models.Workspace]:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Workspace listing not implemented yet."
    )


@router.post("/", response_model=WorkspaceOut, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> models.Workspace:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Workspace creation not implemented yet."
    )
