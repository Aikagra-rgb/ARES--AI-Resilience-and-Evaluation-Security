"""
Runtime Enforcement Gateway Routes.
Provides live prompt inspection, OpenAI-compatible chat guardrail proxy, and status endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, Request

from ares.api.dependencies import verify_api_key
from ares.api.schemas import (
    ChatProxyRequest,
    ChatProxyResponse,
    EnforceRequest,
    EnforceResponse,
    GatewayStatusResponse,
)
from ares.config import Settings, settings as default_settings
from ares.gateway.context import TenantContext
from ares.gateway.enforcer import RuntimeEnforcer
from ares.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gateway", tags=["gateway"])

_shared_enforcer: Optional[RuntimeEnforcer] = None
_shared_store: Optional[QdrantStore] = None


def get_enforcer(request: Request) -> RuntimeEnforcer:
    """
    Return the shared RuntimeEnforcer singleton.
    Loads models, benign reference vectors, and Qdrant connections once,
    preventing memory bloat and cold-start latency across concurrent requests.
    """
    global _shared_enforcer
    # Check if app.state has enforcer or fallback to global singleton
    if hasattr(request.app.state, "enforcer"):
        return request.app.state.enforcer
    if _shared_enforcer is None:
        _shared_enforcer = RuntimeEnforcer(settings=default_settings)
    return _shared_enforcer


def get_qdrant_store(request: Request) -> QdrantStore:
    """Return the shared QdrantStore instance."""
    global _shared_store
    if hasattr(request.app.state, "qdrant_store"):
        return request.app.state.qdrant_store
    if _shared_store is None:
        _shared_store = QdrantStore(settings=default_settings)
    return _shared_store


def get_tenant_context(
    request: Request,
    x_organization_id: Optional[str] = Header(None, alias="X-Ares-Organization-Id"),
    x_project_id: Optional[str] = Header(None, alias="X-Ares-Project-Id"),
    x_canary_token: Optional[str] = Header(None, alias="X-Ares-Canary-Token"),
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> TenantContext:
    """
    Extract multi-tenant execution context from request headers and auth context.
    """
    org_id = (
        x_organization_id
        or getattr(request.state, "organization_id", None)
        or auth_ctx.get("organization_id")
        or "default"
    )
    return TenantContext(
        organization_id=org_id,
        project_id=x_project_id or getattr(request.state, "project_id", None) or auth_ctx.get("project_id"),
        canary_token=x_canary_token,
    )


@router.post("/enforce", response_model=EnforceResponse)
async def enforce_prompt(
    request: EnforceRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    tenant_ctx: TenantContext = Depends(get_tenant_context),
    enforcer: RuntimeEnforcer = Depends(get_enforcer),
) -> EnforceResponse:
    """
    Real-time inspection of an incoming prompt.
    Evaluates Layer-1 Heuristics and Layer-2 Qdrant Vector Similarity Shield (3,695 vectors),
    returning decision: ALLOW, SANITIZE, or BLOCK.
    """
    return await enforcer.enforce(request, context=tenant_ctx)


@router.post("/chat", response_model=ChatProxyResponse)
async def chat_proxy(
    request: ChatProxyRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    tenant_ctx: TenantContext = Depends(get_tenant_context),
    enforcer: RuntimeEnforcer = Depends(get_enforcer),
) -> ChatProxyResponse:
    """
    Guardrail chat proxy for production LLM calls.
    Inspects messages against attack vectors. If blocked, returns immediate refusal.
    If sanitized, wraps prompt with token-efficient boundary before forwarding to downstream LLM.
    """
    messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
    decision, response_text = await enforcer.proxy_chat(
        messages=messages_payload,
        model=request.model,
        application_name=request.application_name,
        context=tenant_ctx,
    )

    return ChatProxyResponse(
        action_taken=decision.action,
        risk_score=decision.risk_score,
        content=response_text,
        model=request.model,
        latency_ms=decision.latency_ms,
        correlation_id=decision.correlation_id,
    )


@router.get("/status", response_model=GatewayStatusResponse)
async def get_gateway_status(
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    store: QdrantStore = Depends(get_qdrant_store),
) -> GatewayStatusResponse:
    """
    Returns active gateway engine health, threshold configs, and Qdrant cluster point count.
    """
    points_count = store.count(store.collection_name)

    return GatewayStatusResponse(
        status="ACTIVE",
        qdrant_points=points_count,
        block_threshold=default_settings.gateway_block_threshold,
        sanitize_threshold=default_settings.gateway_sanitize_threshold,
        canary_token_configured=bool(default_settings.gateway_canary_token),
        version="1.0.0",
    )
