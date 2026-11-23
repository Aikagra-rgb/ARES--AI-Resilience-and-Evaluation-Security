"""
ARES Contextual Multi-Tenancy Container.

Defines the TenantContext model passed into the RuntimeEnforcer singleton
per-request, eliminating the need to allocate heavyweight enforcer instances
per tenant or request. Also provides target-aware application-to-tenant routing
via APPLICATION_TENANT_REGISTRY and resolve_tenant_context().
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional


@dataclass
class TenantContext:
    """
    Contextual parameter container for tenant-specific gateway enforcement.

    Attributes:
        organization_id: Organization / tenant identifier.
        project_id: Optional sub-project or application identifier.
        canary_token: Optional tenant-specific canary token to trap extractions.
        block_threshold: Optional custom vector similarity block threshold (overrides global setting).
        sanitize_threshold: Optional custom vector similarity sanitize threshold (overrides global setting).
        collection_name: Optional custom vector collection name for isolated tenant vectors.
        metadata: Arbitrary tenant metadata dictionary for logging and audit trails.
    """

    organization_id: str = "default"
    project_id: Optional[str] = None
    canary_token: Optional[str] = None
    block_threshold: Optional[float] = None
    sanitize_threshold: Optional[float] = None
    collection_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _clone_context(ctx: TenantContext) -> TenantContext:
    """Return a shallow copy of TenantContext with an isolated metadata dict."""
    return replace(ctx, metadata=dict(ctx.metadata))


_FINTECH_CONTEXT = TenantContext(
    organization_id="org_acme_fintech",
    project_id="AcmeFintechRAG",
    canary_token="CANARY_ACME_FINTECH_77",
    block_threshold=0.80,
    sanitize_threshold=0.65,
    collection_name="org_acme_fintech_attacks",
    metadata={"domain": "fintech", "target_name": "AcmeFintechRAG"},
)

_AGENT_CONTEXT = TenantContext(
    organization_id="org_dev_ops_agent",
    project_id="DevOpsToolAgent",
    canary_token="CANARY_DEVOPS_AGENT_88",
    block_threshold=0.80,
    sanitize_threshold=0.65,
    collection_name="org_dev_ops_agent_attacks",
    metadata={"domain": "devops_agent", "target_name": "DevOpsToolAgent"},
)

_HEALTH_CONTEXT = TenantContext(
    organization_id="org_medshield_health",
    project_id="MedShieldTriage",
    canary_token="CANARY_MEDSHIELD_HIPAA_99",
    block_threshold=0.78,
    sanitize_threshold=0.64,
    collection_name="org_medshield_health_attacks",
    metadata={"domain": "healthcare", "target_name": "MedShieldTriage"},
)

APPLICATION_TENANT_REGISTRY: Dict[str, TenantContext] = {
    "AcmeFintechRAG": _FINTECH_CONTEXT,
    "org_acme_fintech": _FINTECH_CONTEXT,
    "DevOpsToolAgent": _AGENT_CONTEXT,
    "org_dev_ops_agent": _AGENT_CONTEXT,
    "MedShieldTriage": _HEALTH_CONTEXT,
    "org_medshield_health": _HEALTH_CONTEXT,
}


def register_tenant_context(key: str, tenant_ctx: TenantContext) -> None:
    """Register or update a dynamic tenant configuration by application_name or org_id."""
    APPLICATION_TENANT_REGISTRY[key] = _clone_context(tenant_ctx)


def resolve_tenant_context(
    application_name: Optional[str] = None,
    context: Optional[TenantContext] = None,
    default_org_id: str = "default",
) -> TenantContext:
    """Resolve effective TenantContext from explicit overrides, application_name, or registry."""
    base: Optional[TenantContext] = None
    if application_name and application_name in APPLICATION_TENANT_REGISTRY:
        base = APPLICATION_TENANT_REGISTRY[application_name]
    elif context and context.organization_id in APPLICATION_TENANT_REGISTRY:
        base = APPLICATION_TENANT_REGISTRY[context.organization_id]
    elif default_org_id in APPLICATION_TENANT_REGISTRY:
        base = APPLICATION_TENANT_REGISTRY[default_org_id]

    if context is None:
        if base is not None:
            return _clone_context(base)
        return TenantContext(organization_id=default_org_id, project_id=application_name)

    if base is None:
        return _clone_context(context)

    merged_org = (
        context.organization_id
        if context.organization_id and context.organization_id != "default"
        else base.organization_id
    )
    return TenantContext(
        organization_id=merged_org,
        project_id=context.project_id if context.project_id is not None else base.project_id,
        canary_token=context.canary_token if context.canary_token is not None else base.canary_token,
        block_threshold=(
            context.block_threshold if context.block_threshold is not None else base.block_threshold
        ),
        sanitize_threshold=(
            context.sanitize_threshold
            if context.sanitize_threshold is not None
            else base.sanitize_threshold
        ),
        collection_name=(
            context.collection_name if context.collection_name is not None else base.collection_name
        ),
        metadata={**base.metadata, **context.metadata},
    )
