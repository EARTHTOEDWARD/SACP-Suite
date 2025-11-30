"""Job service stubs."""

from __future__ import annotations

from sqlalchemy.orm import Session

from sacp_suite.db import models


def create_simulation_job(db: Session, user: models.User | None, req) -> models.Job:
    # Placeholder: return an in-memory Job object; no persistence yet.
    job = models.Job(
        id=None,  # type: ignore[arg-type]
        experiment_id=req.experiment_id or "00000000-0000-0000-0000-000000000000",
        kind=models.JobKind.SIMULATION,
        status=models.JobStatus.SUCCEEDED,
        payload_json=req.dict(),
        result_uri=None,
        metrics_json=None,
    )
    return job


def get_job(db: Session, job_id: str, user: models.User | None) -> models.Job | None:  # noqa: ARG001
    return None
