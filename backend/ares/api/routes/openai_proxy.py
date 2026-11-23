"""
Drop-In OpenAI-Compatible Reverse Proxy Routes (`/v1/models` & `/v1/chat/completions`).

Allows any standard OpenAI SDK client (`openai.OpenAI(base_url="http://localhost:8000/v1", api_key=...)`)
to route LLM traffic transparently through ARES's multi-layer RuntimeEnforcer, optional per-client
UserPromptOptimizer, and two-way outbound canary/PII redaction firewall.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field

from ares.api.dependencies import enforce_tenant_access, verify_api_key
from ares.api.routes.gateway import get_enforcer
from ares.api.routes.models import _MODELS
from ares.api.schemas import AiProviderEnum
from ares.config import settings as default_settings
from ares.gateway.context import APPLICATION_TENANT_REGISTRY, TenantContext
from ares.gateway.enforcer import RuntimeEnforcer, classify_upstream_llm_error
from ares.llm.models import LLMError
from ares.security.circuit_breaker import CircuitBreakerOpenError
from ares.training.model_registry import ModelRegistry, validate_org_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-proxy"])

_PROVIDER_MAP = {
    AiProviderEnum.openai: "openai",
    AiProviderEnum.groq: "groq",
    AiProviderEnum.gemini: "gemini",
    AiProviderEnum.nvidia_nim: "nvidia",
    AiProviderEnum.ollama: "ollama",
}

_TRUTHY_HEADER_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message payload (supports null content on assistant tool calls)."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(..., min_length=1, max_length=64)
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible `/v1/chat/completions` request payload."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="llama-3.1-8b-instant", min_length=1, max_length=200)
    messages: List[OpenAIChatMessage] = Field(..., min_length=1, max_length=128)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32768)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    user: Optional[str] = None


def _parse_bool_header(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY_HEADER_VALUES


def _build_openai_upstream_error_response(
    error_info: Dict[str, Any],
    ares_headers: Dict[str, str],
    fallback_message: str = "Upstream LLM provider error.",
) -> JSONResponse:
    err_status = int(error_info.get("status_code") or status.HTTP_502_BAD_GATEWAY)
    if err_status not in (status.HTTP_502_BAD_GATEWAY, status.HTTP_503_SERVICE_UNAVAILABLE):
        err_status = status.HTTP_502_BAD_GATEWAY
    return JSONResponse(
        status_code=err_status,
        content={
            "error": {
                "message": str(error_info.get("message") or fallback_message),
                "type": str(error_info.get("type") or "upstream_error"),
                "param": None,
                "code": str(error_info.get("code") or "upstream_llm_error"),
            }
        },
        headers=ares_headers,
    )


def _infer_provider_for_model(model_id: str) -> Optional[str]:
    """Infer ARES LLMProvider string from a catalog or OpenAI-format model ID."""
    for entry in _MODELS:
        if entry.id == model_id:
            return _PROVIDER_MAP.get(entry.provider, entry.provider.value)
    lower = model_id.lower()
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if lower.startswith("gemini-"):
        return "gemini"
    if lower.startswith("nvidia/"):
        return "nvidia"
    if ":" in model_id and not model_id.startswith("ares/"):
        return "ollama"
    return None


@router.get("/models")
async def list_openai_models(
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    OpenAI-compatible `GET /v1/models` catalog backed by ARES's `_MODELS` registry
    and configured client tenant targets (filtered to caller's tenant scope).
    """
    now_ts = int(time.time())
    seen_ids: set[str] = set()
    data: List[Dict[str, Any]] = []

    for m in _MODELS:
        if m.id in seen_ids:
            continue
        seen_ids.add(m.id)
        data.append(
            {
                "id": m.id,
                "object": "model",
                "created": now_ts,
                "owned_by": m.provider.value,
                "permission": [],
                "root": m.id,
                "parent": None,
                "ares_metadata": {
                    "display_name": m.display_name,
                    "context_window": m.context_window,
                    "is_available": m.is_available,
                    "provider": m.provider.value,
                },
            }
        )

    # Ensure configured OpenAI default model is discoverable for drop-in OpenAI SDK clients
    openai_default = default_settings.openai_default_model
    if openai_default and openai_default not in seen_ids:
        seen_ids.add(openai_default)
        data.append(
            {
                "id": openai_default,
                "object": "model",
                "created": now_ts,
                "owned_by": "openai",
                "permission": [],
                "root": openai_default,
                "parent": None,
                "ares_metadata": {
                    "display_name": openai_default,
                    "context_window": 128000,
                    "is_available": bool(default_settings.openai_api_key),
                    "provider": "openai",
                },
            }
        )

    # Include configured client targets (strictly tenant-scoped when non-admin key is used)
    scoped_org = auth_ctx.get("organization_id") if not auth_ctx.get("is_admin") else None
    try:
        clients = ModelRegistry().list_clients()
    except Exception:
        clients = {}

    for org_id, cfg in sorted(clients.items()):
        if scoped_org and org_id != scoped_org:
            continue

        # Virtual tenant-routed model alias: `ares/<org_id>`
        alias_id = f"ares/{org_id}"
        if alias_id not in seen_ids:
            seen_ids.add(alias_id)
            data.append(
                {
                    "id": alias_id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": f"ares-tenant:{org_id}",
                    "permission": [],
                    "root": cfg.get("target_model") or default_settings.groq_default_model,
                    "parent": None,
                    "ares_metadata": {
                        "org_id": org_id,
                        "display_name": cfg.get("display_name") or org_id,
                        "domain": cfg.get("domain") or "general",
                        "application_name": cfg.get("application_name") or org_id,
                        "target_model": cfg.get("target_model"),
                        "target_endpoint_configured": bool(cfg.get("target_endpoint_url")),
                        "model_version": cfg.get("version") or "untrained",
                    },
                }
            )

        # Also expose explicit client `target_model` if configured and not yet listed
        target_model = (cfg.get("target_model") or "").strip()
        if target_model and target_model not in seen_ids:
            seen_ids.add(target_model)
            data.append(
                {
                    "id": target_model,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": f"ares-client:{org_id}",
                    "permission": [],
                    "root": target_model,
                    "parent": None,
                    "ares_metadata": {
                        "org_id": org_id,
                        "display_name": cfg.get("display_name") or target_model,
                        "domain": cfg.get("domain") or "general",
                        "target_endpoint_configured": bool(cfg.get("target_endpoint_url")),
                    },
                }
            )

    return {"object": "list", "data": data}


@router.post("/chat/completions")
async def openai_chat_completions(
    req: OpenAIChatCompletionRequest,
    request: Request,
    x_ares_optimize: Optional[str] = Header(default=None, alias="X-Ares-Optimize"),
    x_ares_inference_mode: Optional[str] = Header(default=None, alias="X-Ares-Inference-Mode"),
    x_ares_organization_id: Optional[str] = Header(default=None, alias="X-Ares-Organization-Id"),
    x_ares_org_id: Optional[str] = Header(default=None, alias="X-Ares-Org-Id"),
    x_ares_app_name: Optional[str] = Header(default=None, alias="X-Ares-App-Name"),
    x_ares_canary_token: Optional[str] = Header(default=None, alias="X-Ares-Canary-Token"),
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    enforcer: RuntimeEnforcer = Depends(get_enforcer),
) -> Any:
    """
    Drop-in OpenAI-compatible `POST /v1/chat/completions` reverse proxy.

    Enforces:
      1. Tenant isolation via `enforce_tenant_access`
      2. Multi-layer inbound security inspection (`RuntimeEnforcer.enforce`)
      3. Optional per-client `UserPromptOptimizer.optimize_prompt` (`X-Ares-Optimize: true`)
      4. Downstream LLM completion forwarding (`LLMClient.complete`)
      5. Two-way outbound canary & PII redaction (`inspect_outbound_response`)
    """
    # Check if caller addressed a virtual `ares/<org_id>` model alias
    model_org_id: Optional[str] = None
    requested_model = req.model.strip()
    if requested_model.startswith("ares/"):
        candidate_org = requested_model.split("/", 1)[1].strip()
        if not candidate_org:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Virtual model alias 'ares/' must specify an organization ID (e.g. 'ares/org_acme_fintech').",
            )
        try:
            model_org_id = validate_org_id(candidate_org)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    # Resolve & strictly enforce tenant scope across header, state, app name, and model alias
    header_org = x_ares_organization_id or x_ares_org_id or getattr(request.state, "organization_id", None)
    if header_org is not None and header_org != "default":
        try:
            header_org = validate_org_id(header_org)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
    if header_org and model_org_id and header_org != model_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Conflicting tenant identifiers: header '{header_org}' does not match "
                f"model alias organization '{model_org_id}'."
            ),
        )

    requested_org = header_org or model_org_id
    scoped_org = enforce_tenant_access(auth_ctx, requested_org)

    if x_ares_app_name and auth_ctx.get("organization_id") and not auth_ctx.get("is_admin"):
        registered_app = APPLICATION_TENANT_REGISTRY.get(x_ares_app_name)
        if registered_app is not None:
            enforce_tenant_access(auth_ctx, registered_app.organization_id)

    effective_org = scoped_org or auth_ctx.get("organization_id") or "default"

    client_cfg: Dict[str, Any] = {}
    if effective_org and effective_org != "default":
        try:
            client_cfg = ModelRegistry().get_client_config(effective_org) or {}
        except Exception:
            client_cfg = {}

    application_name = (
        x_ares_app_name
        or str(client_cfg.get("application_name") or "")
        or (effective_org if effective_org != "default" else "ARES OpenAI Proxy")
    )
    canary_token = x_ares_canary_token or client_cfg.get("canary_token")

    tenant_ctx = TenantContext(
        organization_id=effective_org,
        project_id=application_name,
        canary_token=canary_token,
        metadata={"domain": str(client_cfg.get("domain") or "general"), "proxy": "openai_v1"},
    )

    # Resolve target downstream model & provider
    if model_org_id:
        downstream_model = str(client_cfg.get("target_model") or default_settings.groq_default_model)
    else:
        downstream_model = requested_model

    inferred_provider = _infer_provider_for_model(downstream_model)
    optimize_flag = _parse_bool_header(x_ares_optimize, default=False)
    inference_mode = (x_ares_inference_mode or "auto").strip() or "auto"
    stop_list = [req.stop] if isinstance(req.stop, str) else req.stop

    messages_payload = [m.model_dump(exclude_unset=True) for m in req.messages]

    try:
        decision, response_text, meta = await enforcer.proxy_chat(
            messages=messages_payload,
            model=downstream_model,
            application_name=application_name,
            context=tenant_ctx,
            optimize=optimize_flag,
            inference_mode=inference_mode,
            temperature=req.temperature if req.temperature is not None else 0.7,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stop=stop_list,
            provider=inferred_provider,
            return_metadata=True,
        )
    except (CircuitBreakerOpenError, LLMError, httpx.HTTPError, TimeoutError, ConnectionError) as exc:
        err_info = classify_upstream_llm_error(
            exc,
            str(inferred_provider or default_settings.victim_provider),
        )
        fallback_headers = {
            "X-Ares-Action": "ALLOW",
            "X-Ares-Risk-Score": "0",
            "X-Ares-Org-Id": effective_org,
            "X-Ares-Token-Savings-Pct": "0.00",
            "X-Ares-Correlation-Id": uuid.uuid4().hex[:12],
        }
        return _build_openai_upstream_error_response(
            err_info,
            fallback_headers,
            fallback_message=str(err_info.get("message") or "Upstream LLM provider error."),
        )

    created_ts = int(time.time())
    completion_id = f"chatcmpl-{decision.correlation_id}"
    resolved_org = str(meta.get("org_id") or effective_org)
    savings_pct = float(meta.get("token_savings_pct") or 0.0)
    finish_reason = str(meta.get("finish_reason") or "stop")
    model_used = str(meta.get("model") or downstream_model)
    usage = dict(meta.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    ares_headers = {
        "X-Ares-Action": decision.action.value,
        "X-Ares-Risk-Score": str(decision.risk_score),
        "X-Ares-Org-Id": resolved_org,
        "X-Ares-Token-Savings-Pct": f"{savings_pct * 100.0:.2f}",
        "X-Ares-Correlation-Id": decision.correlation_id,
    }

    upstream_error = meta.get("upstream_error")
    if isinstance(upstream_error, dict):
        return _build_openai_upstream_error_response(
            upstream_error,
            ares_headers,
            fallback_message=response_text,
        )

    if req.stream:
        async def _sse_chunks():
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_used,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_used,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _sse_chunks(),
            media_type="text/event-stream",
            headers=ares_headers,
        )

    payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_used,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "ares": {
            "action": decision.action.value,
            "risk_score": decision.risk_score,
            "severity": decision.severity.value,
            "matched_technique": decision.matched_technique,
            "org_id": resolved_org,
            "token_savings_pct": round(savings_pct, 4),
            "optimization_applied": bool(meta.get("optimization_applied")),
            "outbound_redactions": list(meta.get("outbound_redactions") or []),
            "correlation_id": decision.correlation_id,
        },
    }
    return JSONResponse(content=payload, headers=ares_headers)
