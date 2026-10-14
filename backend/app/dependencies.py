from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Organization, Project, User
from ares.api.dependencies import is_valid_api_key


@dataclass(frozen=True)
class CurrentUser:
    id: str
    organization_id: str
    email: str


def _extract_token(x_ares_api_key: str | None, authorization: str | None) -> str | None:
    if x_ares_api_key:
        return x_ares_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _normalise_email(raw: str) -> str:
    email = raw.strip().lower()
    if not email or len(email) > 320 or "@" not in email or any(c.isspace() for c in email):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid caller identity.")
    return email


def get_current_user(
    db: Session = Depends(get_db),
    x_ares_user_email: str | None = Header(default=None),
    x_ares_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> CurrentUser:
    """
    Resolve the calling user.

    * Valid API key  -> the key authenticates the calling application; `X-Ares-User-Email`
      names the end user on whose behalf it acts (trusted-frontend model).
    * No/invalid key -> only accepted when ALLOW_DEV_IDENTITY=true in a development
      environment (enforced at settings load). Otherwise 401.

    The email header alone is never sufficient to authenticate a caller outside development.
    """
    settings = get_settings()
    token = _extract_token(x_ares_api_key, authorization)

    if is_valid_api_key(token):
        email = _normalise_email(x_ares_user_email or "service@ares.local")
    elif token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Invalid ARES API Key.")
    elif settings.allow_dev_identity and settings.is_development:
        email = _normalise_email(x_ares_user_email or "developer@local")
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    organization = db.scalar(select(Organization).where(Organization.name == "Local development"))
    if organization is None:
        organization = Organization(name="Local development")
        db.add(organization)
        db.flush()
    user = db.scalar(select(User).where(User.organization_id == organization.id, User.email == email))
    if user is None:
        user = User(organization_id=organization.id, email=email, display_name=email.split("@")[0])
        db.add(user)
        db.flush()

    default_project = db.scalar(
        select(Project).where(Project.organization_id == organization.id, Project.name == "Default project")
    )
    if default_project is None:
        db.add(
            Project(
                organization_id=organization.id,
                created_by_user_id=user.id,
                name="Default project",
                description="Local development project",
            )
        )
    db.commit()
    return CurrentUser(id=user.id, organization_id=organization.id, email=user.email)
