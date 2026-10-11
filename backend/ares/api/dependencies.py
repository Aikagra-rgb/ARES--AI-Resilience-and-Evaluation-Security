"""
ARES API Security & Authentication Dependencies.

Supports API Key authentication via `X-Ares-Api-Key` or `Authorization: Bearer <key>`.

Auth is enabled by default. `ARES_AUTH_ENABLED=false` is honoured only in a development/test
environment (see `ares.config.Settings._enforce_production_security`), so a production
deployment can never silently run with open routes.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import HTTPException, Header, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from ares.config import settings

api_key_header = APIKeyHeader(name="X-Ares-Api-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def is_valid_api_key(token: Optional[str]) -> bool:
    """Constant-time comparison against the configured API key."""
    if not token:
        return False
    return hmac.compare_digest(settings.api_key.encode("utf-8"), token.encode("utf-8"))


async def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
    x_ares_user_email: Optional[str] = Header(None, alias="X-Ares-User-Email"),
) -> dict:
    """
    Verify the caller's API key or bearer token.

    The `X-Ares-User-Email` header is only trusted when it accompanies a valid key (the key
    authenticates the calling application; the header names the end user on whose behalf it acts).
    """
    token = x_api_key
    if not token and bearer:
        token = bearer.credentials

    key_ok = is_valid_api_key(token)

    if settings.auth_enabled:
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Authentication credentials were not provided. "
                    "Include 'X-Ares-Api-Key' or 'Authorization: Bearer <token>'."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not key_ok:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Invalid ARES API Key.",
            )

    trusted_identity = key_ok or not settings.auth_enabled
    return {
        "authenticated": key_ok,
        "user_email": (x_ares_user_email if trusted_identity and x_ares_user_email else "system@ares.local"),
        "role": "admin" if key_ok else "anonymous",
    }
