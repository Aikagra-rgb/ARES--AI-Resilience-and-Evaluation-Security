"""Structured per-target Gateway decision telemetry and audit ring buffer.

Captures `(organization_id, target_name, attack_category, decision, latency_ms)` for
observability, red-team verification assertions, and Phase 4 per-client model training.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ares.api.schemas import EnforceRequest, EnforceResponse
from ares.observability.logging import set_correlation_id, set_organization_id

logger = logging.getLogger("ares.gateway.decision")

_DECISION_LOG_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=2000)


def log_gateway_decision(
    decision: EnforceResponse,
    request: EnforceRequest,
    organization_id: str,
    target_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit a structured Gateway decision record to logs and in-memory telemetry buffer."""
    effective_target = target_name or request.application_name
    attack_category = decision.matched_technique or "clean"
    entry: Dict[str, Any] = {
        "timestamp": decision.timestamp.isoformat(),
        "correlation_id": decision.correlation_id,
        "organization_id": organization_id,
        "target_name": effective_target,
        "attack_category": attack_category,
        "decision": decision.action.value,
        "severity": decision.severity.value,
        "risk_score": decision.risk_score,
        "vector_similarity": decision.vector_similarity or 0.0,
        "latency_ms": decision.latency_ms,
        "prompt_preview": request.prompt[:200],
    }
    if extra_metadata:
        entry["metadata"] = dict(extra_metadata)

    set_correlation_id(decision.correlation_id)
    set_organization_id(organization_id)

    logger.info(
        "gateway_decision org=%s target=%s category=%s decision=%s severity=%s latency_ms=%.2f",
        organization_id,
        effective_target,
        attack_category,
        decision.action.value,
        decision.severity.value,
        decision.latency_ms,
        extra=entry,
    )

    _DECISION_LOG_BUFFER.append(entry)
    return entry


def get_recent_gateway_logs(
    target_name: Optional[str] = None,
    organization_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return recent structured decision records filtered by target_name or organization_id."""
    records = list(_DECISION_LOG_BUFFER)
    if target_name is not None:
        records = [r for r in records if r.get("target_name") == target_name]
    if organization_id is not None:
        records = [r for r in records if r.get("organization_id") == organization_id]
    if limit <= 0:
        return []
    return list(reversed(records))[:limit]


def clear_gateway_logs() -> None:
    """Clear the in-memory decision log buffer (primarily for isolated tests)."""
    _DECISION_LOG_BUFFER.clear()
