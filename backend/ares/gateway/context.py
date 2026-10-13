"""
ARES Contextual Multi-Tenancy Container.

Defines the TenantContext model passed into the RuntimeEnforcer singleton
per-request, eliminating the need to allocate heavyweight enforcer instances
per tenant or request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
