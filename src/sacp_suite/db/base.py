"""Database base, engine, and session helpers."""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Default to a local SQLite DB for development; override with SACP_DB_URL.
DB_URL = os.getenv("SACP_DB_URL", "sqlite:///./var/sacp.db")

engine = create_engine(DB_URL, future=True)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
