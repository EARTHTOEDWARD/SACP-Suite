"""Auth endpoints (stubs)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sacp_suite.db.base import get_db

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str | None = None


@router.post("/signup", response_model=Token)
def signup_user(payload: UserCreate, db: Session = Depends(get_db)) -> Token:  # noqa: ARG001
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Signup not implemented yet."
    )


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:  # noqa: ARG001
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Login not implemented yet."
    )
