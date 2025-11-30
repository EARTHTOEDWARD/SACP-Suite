"""Job endpoints for simulations/analyses (stubs)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sacp_suite.api.deps import get_current_user, get_db_session
from sacp_suite.db import models
from sacp_suite.db.models import JobKind, JobStatus
from sacp_suite.services import jobs as job_service

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


class SimulationRequest(BaseModel):
    workspace_id: str
    experiment_id: str | None = None
    dynamics_type: str
    params: dict
    T: float
    dt: float


class JobOut(BaseModel):
    id: str
    experiment_id: str
    kind: JobKind
    status: JobStatus

    class Config:
        orm_mode = True


@router.post("/simulate", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def enqueue_simulation(
    payload: SimulationRequest,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> models.Job:
    job = job_service.create_simulation_job(db=db, user=current_user, req=payload)
    return job


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    db: Session = Depends(get_db_session),
    current_user: models.User = Depends(get_current_user),  # noqa: ARG001
) -> models.Job:
    job = job_service.get_job(db, job_id=job_id, user=current_user)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
