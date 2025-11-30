"""Shared API dependencies: DB session and auth placeholder."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from sacp_suite.db.base import get_db
from sacp_suite.db import models


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def get_current_user(db: Session = Depends(get_db_session)) -> models.User:
    # TODO: replace with real JWT/OAuth2 lookup.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Authentication not yet implemented.",
    )
