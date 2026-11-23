"""
ARES API Security & Authentication Dependencies.

Supports API Key authentication via `X-Ares-Api-Key` or `Authorization: Bearer <key>`.

Auth is enabled by default. `ARES_AUTH_ENABLED=false` is honoured only in a development/test
environment (see `ares.config.Settings._enforce_production_security`), so a production
deployment can never silently run with open routes.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Dict, Optional

from fastapi import HTTPException, Header, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from ares.config import settings
from ares.training.model_registry import ModelRegistry

api_key_header = APIKeyHeader(name="X-Ares-Api-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def hash_api_key(token: str) -> str:
    """Compute SHA-256 hex digest of an API key for storage lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def resolve_api_key_identity(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Resolve an API key to either the master admin identity or a tenant-scoped identity.
    Checks the master `settings.api_key`, env-configured `ARES_TENANT_API_KEYS`,
    and hashed per-client API keys stored in `ModelRegistry`.
    """
    if not token:
        return None

    token_bytes = token.encode("utf-8")
    if hmac.compare_digest(settings.api_key.encode("utf-8"), token_bytes):
        return {
            "role": "admin",
            "organization_id": None,
            "is_admin": True,
        }

    matched_env_org: Optional[str] = None
    for oid, configured_secret in settings.parsed_tenant_api_keys.items():
        if hmac.compare_digest(configured_secret.encode("utf-8"), token_bytes):
            matched_env_org = oid
    if matched_env_org:
        return {
            "role": "tenant",
            "organization_id": matched_env_org,
            "is_admin": False,
        }

    try:
        registry_org = ModelRegistry().find_org_by_api_key_hash(hash_api_key(token))
        if registry_org:
            return {
                "role": "tenant",
                "organization_id": registry_org,
                "is_admin": False,
            }
    except Exception:
        pass

    return None


def is_valid_api_key(token: Optional[str]) -> bool:
    """Constant-time check against admin or tenant-scoped API keys."""
    return resolve_api_key_identity(token) is not None


async def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
    x_ares_user_email: Optional[str] = Header(None, alias="X-Ares-User-Email"),
) -> dict:
    """
    Verify the caller's API key or bearer token and attach tenant scope metadata.

    The `X-Ares-User-Email` header is only trusted when it accompanies a valid key (the key
    authenticates the calling application; the header names the end user on whose behalf it acts).
    """
    token = x_api_key
    if not token and bearer:
        token = bearer.credentials

    identity = resolve_api_key_identity(token)
    key_ok = identity is not None

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
    if identity is not None:
        role = identity["role"]
        org_id = identity["organization_id"]
        is_admin = bool(identity["is_admin"])
    else:
        role = "anonymous"
        org_id = None
        is_admin = True

    return {
        "authenticated": key_ok,
        "user_email": (x_ares_user_email if trusted_identity and x_ares_user_email else "system@ares.local"),
        "role": role,
        "organization_id": org_id,
        "is_admin": is_admin,
    }


def enforce_tenant_access(
    auth_ctx: Optional[Dict[str, Any]],
    target_org_id: Optional[str],
    *,
    allow_all_for_tenant: bool = False,
) -> Optional[str]:
    """
    Enforce tenant isolation for scoped API keys.
    Returns the effective `org_id` for the request or raises 403 on cross-tenant access.
    """
    if not auth_ctx or auth_ctx.get("is_admin") or not auth_ctx.get("organization_id"):
        return target_org_id

    scoped_org = str(auth_ctx["organization_id"])
    if target_org_id is None or target_org_id == "default":
        return scoped_org
    if allow_all_for_tenant and target_org_id == "all":
        return scoped_org
    if target_org_id != scoped_org:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key is scoped to organization '{scoped_org}' and cannot access '{target_org_id}'.",
        )
    return scoped_org
