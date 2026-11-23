"""
ARES Client Onboarding, Per-Client Model Registry, and Partner Deployment API Routes.

Exposes REST endpoints for registering enterprise tenant organizations,
inspecting per-client DistilBERT/hybrid intent & threat classifiers and
compression adapters, triggering per-client retraining, and exporting
turnkey partner sidecar deployment bundles.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ares.api.dependencies import enforce_tenant_access, hash_api_key, verify_api_key
from ares.gateway.context import (
    APPLICATION_TENANT_REGISTRY,
    TenantContext,
    register_tenant_context,
    resolve_tenant_context,
)
from ares.gateway.logging import get_recent_gateway_logs
from ares.training.client_model_trainer import export_consolidated_ml_stats, train_client_models
from ares.training.model_registry import (
    DEFAULT_CLIENT_PROFILES,
    MODELS_ROOT_DIR,
    ModelRegistry,
    validate_org_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["clients"])


def _validate_org_id_or_400(org_id: str) -> str:
    try:
        return validate_org_id(org_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


class ClientCreateRequest(BaseModel):
    """Payload to register or update an ARES client organization."""

    org_id: str = Field(..., min_length=2, max_length=128, description="Unique organization/tenant identifier")
    display_name: Optional[str] = Field(default=None, description="Human-readable organization name")
    domain: str = Field(default="general", description="Target domain (fintech, software, healthcare, general)")
    application_name: Optional[str] = Field(default=None, description="Registered application header/identifier")
    canary_token: Optional[str] = Field(default=None, description="Unique exfiltration trap canary token")
    block_threshold: float = Field(default=0.75, ge=0.1, le=1.0)
    sanitize_threshold: float = Field(default=0.60, ge=0.1, le=1.0)
    collection_name: Optional[str] = Field(default=None)
    fidelity_thresholds: Optional[Dict[str, float]] = Field(default=None)
    auto_train: bool = Field(default=False, description="Trigger immediate per-client model training on registration")
    target_endpoint_url: Optional[str] = Field(default=None, description="Optional staging LLM HTTP endpoint URL for live external red-teaming")
    target_auth_header: Optional[str] = Field(default=None, description="Optional HTTP header ('Authorization: Bearer ...' or 'Header: Value') for target endpoint")
    target_request_format: Optional[str] = Field(default="openai_chat", description="External target payload format ('openai_chat' or 'json_query')")
    target_model: Optional[str] = Field(default=None, description="Model name passed to external OpenAI-compatible target endpoints")


class ClientSummaryResponse(BaseModel):
    """Unified tenant + trained model status view."""

    org_id: str
    display_name: str
    domain: str
    application_name: str
    canary_configured: bool
    canary_token_preview: Optional[str] = None
    block_threshold: float
    sanitize_threshold: float
    model_version: str
    intent_model: Optional[str] = None
    compression_adapter: Optional[str] = None
    fidelity_thresholds: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recent_gateway_events: int = 0
    recent_blocked_events: int = 0
    target_endpoint_configured: bool = False
    target_endpoint_url: Optional[str] = None
    target_request_format: str = "openai_chat"
    target_model: Optional[str] = None
    tenant_api_key_configured: bool = False


class ClientApiKeyResponse(BaseModel):
    """Generated tenant-scoped API key returned once upon rotation."""

    org_id: str
    api_key: str
    key_preview: str
    created_at: str


class ClientModelsResponse(BaseModel):
    """Per-client model artifact & evaluation metrics response."""

    org_id: str
    domain: str
    version: str
    intent_model: Optional[str] = None
    compression_adapter: Optional[str] = None
    fidelity_thresholds: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    updated_at: Optional[str] = None


class ClientMLStatsResponse(BaseModel):
    """Detailed per-client ML evaluation statistics (precision, recall, F1, CV accuracy, confusion matrix)."""

    org_id: str
    domain: str
    generated_at: Optional[str] = None
    threat_ml_stats: Dict[str, Any] = Field(default_factory=dict)
    intent_ml_stats: Dict[str, Any] = Field(default_factory=dict)
    compression_stats: Dict[str, Any] = Field(default_factory=dict)


class ClientDeploymentConfigResponse(BaseModel):
    """Partner deployment kit bundle (Docker Compose sidecar, env vars, SDK & OpenAI proxy snippets)."""

    org_id: str
    application_name: str
    domain: str
    canary_token: str
    gateway_headers: Dict[str, str]
    env_vars: Dict[str, str]
    docker_compose_yaml: str
    python_sdk_snippet: str
    openai_proxy_snippet: str
    deploy_command: str


def _mask_canary(canary: Optional[str]) -> Optional[str]:
    if not canary:
        return None
    if len(canary) <= 8:
        return "***"
    return f"{canary[:10]}...{canary[-2:]}"


def _build_client_summary(org_id: str, cfg: Dict[str, Any]) -> ClientSummaryResponse:
    from ares.config import settings

    app_name = str(cfg.get("application_name") or org_id)
    tenant = resolve_tenant_context(application_name=app_name, default_org_id=org_id)
    effective_canary = cfg.get("canary_token") or tenant.canary_token
    logs = get_recent_gateway_logs(limit=200, organization_id=org_id)
    blocked_cnt = sum(1 for item in logs if str(item.get("action", "")).upper() == "BLOCK")
    target_url = cfg.get("target_endpoint_url")
    tenant_key_configured = bool(cfg.get("api_key_hash")) or (org_id in settings.parsed_tenant_api_keys)

    return ClientSummaryResponse(
        org_id=org_id,
        display_name=str(cfg.get("display_name") or org_id),
        domain=str(cfg.get("domain") or "general"),
        application_name=app_name,
        canary_configured=bool(effective_canary),
        canary_token_preview=_mask_canary(effective_canary),
        block_threshold=float(tenant.block_threshold or 0.75),
        sanitize_threshold=float(tenant.sanitize_threshold or 0.60),
        model_version=str(cfg.get("version") or "untrained"),
        intent_model=cfg.get("intent_model"),
        compression_adapter=cfg.get("compression_adapter"),
        fidelity_thresholds=dict(cfg.get("fidelity_thresholds") or {}),
        metrics=dict(cfg.get("metrics") or {}),
        recent_gateway_events=len(logs),
        recent_blocked_events=blocked_cnt,
        target_endpoint_configured=bool(target_url),
        target_endpoint_url=target_url,
        target_request_format=str(cfg.get("target_request_format") or "openai_chat"),
        target_model=cfg.get("target_model"),
        tenant_api_key_configured=tenant_key_configured,
    )


@router.get("/api/v1/clients", response_model=List[ClientSummaryResponse])
@router.get("/api/clients", response_model=List[ClientSummaryResponse])
async def list_clients(
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> List[ClientSummaryResponse]:
    """List all onboarded clients and reference targets with model and gateway status."""
    registry = ModelRegistry()
    clients_map = registry.list_clients()
    # Also include any dynamically registered tenants in memory that haven't been saved yet
    for key, ctx in APPLICATION_TENANT_REGISTRY.items():
        if ctx.organization_id not in clients_map and key == ctx.organization_id:
            clients_map[ctx.organization_id] = {
                "org_id": ctx.organization_id,
                "display_name": ctx.project_id or ctx.organization_id,
                "domain": ctx.metadata.get("domain", "general"),
                "application_name": ctx.project_id or ctx.organization_id,
                "canary_token": ctx.canary_token,
                "version": "untrained",
                "fidelity_thresholds": {},
                "metrics": {},
            }
    scoped_org = auth_ctx.get("organization_id") if not auth_ctx.get("is_admin") else None
    if scoped_org:
        clients_map = {k: v for k, v in clients_map.items() if k == scoped_org}
    return [_build_client_summary(oid, cfg) for oid, cfg in sorted(clients_map.items())]


@router.post(
    "/api/v1/clients",
    response_model=ClientSummaryResponse,
    status_code=status.HTTP_201_CREATED,
)
@router.post(
    "/api/clients",
    response_model=ClientSummaryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_or_update_client(
    req: ClientCreateRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientSummaryResponse:
    """Register a new client organization (or update existing) and wire TenantContext."""
    org_id = _validate_org_id_or_400(req.org_id)
    enforce_tenant_access(auth_ctx, org_id)
    app_name = req.application_name or org_id
    canary = req.canary_token or f"CANARY_{org_id.upper()}_01"
    collection = req.collection_name or f"{org_id}_attacks"

    tenant_ctx = TenantContext(
        organization_id=org_id,
        project_id=app_name,
        canary_token=canary,
        block_threshold=req.block_threshold,
        sanitize_threshold=req.sanitize_threshold,
        collection_name=collection,
        metadata={"domain": req.domain, "target_name": app_name},
    )
    register_tenant_context(org_id, tenant_ctx)
    register_tenant_context(app_name, tenant_ctx)

    registry = ModelRegistry()
    entry = registry.register_client(
        org_id=org_id,
        display_name=req.display_name or org_id,
        domain=req.domain,
        application_name=app_name,
        canary_token=canary,
        fidelity_thresholds=req.fidelity_thresholds,
        target_endpoint_url=req.target_endpoint_url,
        target_auth_header=req.target_auth_header,
        target_request_format=req.target_request_format,
        target_model=req.target_model,
    )

    if req.auto_train:
        train_res = train_client_models(org_id, domain=req.domain)
        entry = train_res["registry_entry"]

    return _build_client_summary(org_id, entry)


@router.post("/api/v1/clients/{org_id}/api-key", response_model=ClientApiKeyResponse)
@router.post("/api/clients/{org_id}/api-key", response_model=ClientApiKeyResponse)
async def rotate_client_api_key(
    org_id: str,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientApiKeyResponse:
    """Generate and persist a tenant-scoped API key hash for `org_id`."""
    org_id = _validate_org_id_or_400(org_id)
    enforce_tenant_access(auth_ctx, org_id)
    registry = ModelRegistry()
    if org_id not in registry.list_clients():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client organization '{org_id}' is not registered.",
        )

    raw_key = f"ares_tnt_{org_id}_{secrets.token_hex(16)}"
    registry.set_client_api_key_hash(org_id, hash_api_key(raw_key))
    return ClientApiKeyResponse(
        org_id=org_id,
        api_key=raw_key,
        key_preview=f"{raw_key[:14]}...{raw_key[-4:]}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/api/v1/clients/{org_id}/models", response_model=ClientModelsResponse)
@router.get("/api/clients/{org_id}/models", response_model=ClientModelsResponse)
async def get_client_models(
    org_id: str,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientModelsResponse:
    """Inspect per-client trained classifier and compression adapter metadata."""
    org_id = _validate_org_id_or_400(org_id)
    enforce_tenant_access(auth_ctx, org_id)
    registry = ModelRegistry()
    cfg = registry.get_client_config(org_id)
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client organization '{org_id}' is not registered.",
        )
    return ClientModelsResponse(
        org_id=org_id,
        domain=str(cfg.get("domain") or "general"),
        version=str(cfg.get("version") or "untrained"),
        intent_model=cfg.get("intent_model"),
        compression_adapter=cfg.get("compression_adapter"),
        fidelity_thresholds=dict(cfg.get("fidelity_thresholds") or {}),
        metrics=dict(cfg.get("metrics") or {}),
        updated_at=cfg.get("updated_at"),
    )


@router.post("/api/v1/clients/{org_id}/train", response_model=ClientModelsResponse)
@router.post("/api/clients/{org_id}/train", response_model=ClientModelsResponse)
async def trigger_client_training(
    org_id: str,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientModelsResponse:
    """Run per-client DistilBERT/hybrid classifier & compression adapter training for `org_id`."""
    org_id = _validate_org_id_or_400(org_id)
    enforce_tenant_access(auth_ctx, org_id)
    registry = ModelRegistry()
    cfg = registry.get_client_config(org_id)
    domain = cfg.get("domain") if cfg else None
    result = train_client_models(org_id, domain=domain)
    export_consolidated_ml_stats([result])
    entry = result["registry_entry"]
    return ClientModelsResponse(
        org_id=org_id,
        domain=str(entry.get("domain") or "general"),
        version=str(entry.get("version") or "v1.0"),
        intent_model=entry.get("intent_model"),
        compression_adapter=entry.get("compression_adapter"),
        fidelity_thresholds=dict(entry.get("fidelity_thresholds") or {}),
        metrics=dict(entry.get("metrics") or {}),
        updated_at=entry.get("updated_at"),
    )


@router.get("/api/v1/clients/{org_id}/stats", response_model=ClientMLStatsResponse)
@router.get("/api/clients/{org_id}/stats", response_model=ClientMLStatsResponse)
async def get_client_ml_stats(
    org_id: str,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientMLStatsResponse:
    """Return full ML classification statistics (accuracy, CV, precision, recall, F1, confusion matrix) for `org_id`."""
    org_id = _validate_org_id_or_400(org_id)
    enforce_tenant_access(auth_ctx, org_id)
    resolved_root = MODELS_ROOT_DIR.resolve()
    stats_file = (resolved_root / org_id / "intent_classifier" / "ml_stats.json").resolve()
    if resolved_root not in stats_file.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid organization ID '{org_id}'.",
        )
    if not stats_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ML statistics report not generated yet for '{org_id}'. Train the client model first.",
        )

    report = json.loads(stats_file.read_text(encoding="utf-8"))
    adapter_dir = (resolved_root / org_id / "compression_adapter").resolve()
    if resolved_root not in adapter_dir.parents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid organization ID '{org_id}'.",
        )
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    adapter_meta_path = adapter_dir / "ares_adapter_metadata.json"
    compression_stats: Dict[str, Any] = {}
    merged_acfg: Dict[str, Any] = {}
    for cfg_file in (adapter_cfg_path, adapter_meta_path):
        if cfg_file.exists():
            try:
                merged_acfg.update(json.loads(cfg_file.read_text(encoding="utf-8")))
            except Exception:
                pass
    if merged_acfg:
        compression_stats = {
            "sft_pair_count": merged_acfg.get("sft_pair_count", 0),
            "exemplar_count": merged_acfg.get("exemplar_count", 0),
            "observed_avg_token_savings": merged_acfg.get("observed_avg_token_savings", 0.0),
            "target_compression_ratio": merged_acfg.get("target_compression_ratio", 0.0),
        }

    return ClientMLStatsResponse(
        org_id=org_id,
        domain=str(report.get("domain") or "general"),
        generated_at=report.get("generated_at"),
        threat_ml_stats=dict(report.get("threat_ml_stats") or {}),
        intent_ml_stats=dict(report.get("intent_ml_stats") or {}),
        compression_stats=compression_stats,
    )


@router.get("/api/v1/clients/{org_id}/deployment-config", response_model=ClientDeploymentConfigResponse)
@router.get("/api/clients/{org_id}/deployment-config", response_model=ClientDeploymentConfigResponse)
async def get_client_deployment_config(
    org_id: str,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
) -> ClientDeploymentConfigResponse:
    """Return turnkey partner sidecar Docker Compose, env vars, and Python SDK integration config."""
    org_id = _validate_org_id_or_400(org_id)
    enforce_tenant_access(auth_ctx, org_id)
    registry = ModelRegistry()
    cfg = registry.get_client_config(org_id) or (
        {"org_id": org_id, **DEFAULT_CLIENT_PROFILES[org_id]}
        if org_id in DEFAULT_CLIENT_PROFILES
        else None
    )
    if cfg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client organization '{org_id}' is not registered.",
        )

    app_name = str(cfg.get("application_name") or org_id)
    domain = str(cfg.get("domain") or "general")
    canary = str(cfg.get("canary_token") or f"CANARY_{org_id.upper()}_01")
    tenant = resolve_tenant_context(application_name=app_name, default_org_id=org_id)

    env_vars = {
        "ARES_ORG_ID": org_id,
        "ARES_APP_NAME": app_name,
        "ARES_GATEWAY_CANARY_TOKEN": canary,
        "ARES_GATEWAY_BLOCK_THRESHOLD": str(tenant.block_threshold or 0.75),
        "ARES_GATEWAY_SANITIZE_THRESHOLD": str(tenant.sanitize_threshold or 0.60),
        "ARES_PER_CLIENT_INTENT_MODEL": str(cfg.get("intent_model") or f"models/{org_id}/intent_classifier/"),
        "ARES_PER_CLIENT_ADAPTER": str(cfg.get("compression_adapter") or f"models/{org_id}/compression_adapter/"),
    }

    docker_compose_yaml = (
        f"services:\n"
        f"  ares-sidecar-{org_id.replace('_', '-')}:\n"
        f"    image: ares-gateway:latest\n"
        f"    ports:\n"
        f"      - \"8080:8000\"\n"
        f"    environment:\n"
        f"      - ARES_ORG_ID={org_id}\n"
        f"      - ARES_APP_NAME={app_name}\n"
        f"      - ARES_GATEWAY_CANARY_TOKEN={canary}\n"
        f"      - ARES_GATEWAY_BLOCK_THRESHOLD={env_vars['ARES_GATEWAY_BLOCK_THRESHOLD']}\n"
        f"    volumes:\n"
        f"      - ./backend/models/{org_id}:/app/backend/models/{org_id}:ro\n"
    )

    python_sdk_snippet = (
        f"from ares_sdk.client import AresClient\n\n"
        f"async with AresClient(api_key=\"$ARES_API_KEY\", base_url=\"http://localhost:8000\") as ares:\n"
        f"    verdict = await ares.enforce(prompt=user_input, application_name=\"{app_name}\")\n"
        f"    if verdict[\"action\"] == \"ALLOW\":\n"
        f"        optimized = await ares.optimize(prompt=user_input, org_id=\"{org_id}\")\n"
    )

    target_model_hint = str(cfg.get("target_model") or "llama-3.1-8b-instant")
    openai_proxy_snippet = (
        f"import openai\n\n"
        f"client = openai.OpenAI(\n"
        f"    base_url=\"http://localhost:8000/v1\",\n"
        f"    api_key=\"<YOUR_ARES_TENANT_API_KEY>\",\n"
        f"    default_headers={{\n"
        f"        \"X-Ares-Organization-Id\": \"{org_id}\",\n"
        f"        \"X-Ares-App-Name\": \"{app_name}\",\n"
        f"        \"X-Ares-Optimize\": \"true\",\n"
        f"        \"X-Ares-Inference-Mode\": \"client_adapter\",\n"
        f"    }},\n"
        f")\n"
        f"resp = client.chat.completions.create(\n"
        f"    model=\"{target_model_hint}\",\n"
        f"    messages=[{{\"role\": \"user\", \"content\": user_input}}],\n"
        f")\n"
    )

    return ClientDeploymentConfigResponse(
        org_id=org_id,
        application_name=app_name,
        domain=domain,
        canary_token=canary,
        gateway_headers={
            "X-Ares-Api-Key": "<YOUR_ARES_API_KEY>",
            "X-Ares-Organization-Id": org_id,
            "X-Ares-Org-Id": org_id,
            "X-Ares-App-Name": app_name,
        },
        env_vars=env_vars,
        docker_compose_yaml=docker_compose_yaml,
        python_sdk_snippet=python_sdk_snippet,
        openai_proxy_snippet=openai_proxy_snippet,
        deploy_command=f"./deploy_partner.sh --org={org_id} --app={app_name} --api-key=<YOUR_ARES_API_KEY>",
    )
