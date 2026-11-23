"""
API Routes for Attack Moderation Queue and Threat Intelligence Ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ares.api.dependencies import enforce_tenant_access, verify_api_key
from ares.api.routes.gateway import get_qdrant_store
from ares.config import settings as default_settings
from ares.intel.feed_ingester import ThreatIntelIngester
from ares.intel.models import IngestResult
from ares.moderation.feedback_export import export_feedback_to_training
from ares.moderation.models import ModerationItem, ModerationStatus, SubmissionSeverity
from ares.moderation.queue import AttackModerationQueue
from ares.training.model_registry import validate_org_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/moderation", tags=["moderation"])

_shared_queue: Optional[AttackModerationQueue] = None
_shared_ingester: Optional[ThreatIntelIngester] = None


def get_moderation_queue(request: Request) -> AttackModerationQueue:
    """Return the shared AttackModerationQueue singleton."""
    global _shared_queue
    if hasattr(request.app.state, "moderation_queue"):
        return request.app.state.moderation_queue
    if _shared_queue is None:
        store = get_qdrant_store(request)
        _shared_queue = AttackModerationQueue(qdrant_store=store)
    return _shared_queue


def get_threat_ingester(request: Request) -> ThreatIntelIngester:
    """Return the shared ThreatIntelIngester singleton."""
    global _shared_ingester
    if hasattr(request.app.state, "threat_ingester"):
        return request.app.state.threat_ingester
    if _shared_ingester is None:
        queue = get_moderation_queue(request)
        _shared_ingester = ThreatIntelIngester(moderation_queue=queue)
    return _shared_ingester


class SubmitAttackRequest(BaseModel):
    attack_text: str = Field(..., description="The adversarial prompt text")
    category: str = Field(default="jailbreak", description="Attack category")
    severity: str = Field(default="high", description="Severity assessment")
    source: str = Field(default="operator_submission", description="Source or probe identifier")
    domain: str = Field(default="general", description="Target domain")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata tags")
    bypass_moderation: bool = Field(default=False, description="Whether to bypass if clean")


class ReviewDecisionRequest(BaseModel):
    reviewer_id: str = Field(..., description="Identity of the reviewing analyst")
    notes: Optional[str] = Field(None, description="Review justification or analysis notes")


class RejectionRequest(BaseModel):
    reviewer_id: str = Field(..., description="Identity of the reviewing analyst")
    reason: str = Field(..., description="Reason for rejecting the sample")


class IngestFeedRequest(BaseModel):
    feed_type: str = Field(..., description="Type of threat feed (e.g., 'mitre_atlas', 'owasp_llm', 'generic')")
    feed_name: str = Field(default="external_intel", description="Identifier name for the feed")
    data: Any = Field(..., description="Raw feed data object (dictionary or list)")


class ExportTrainingRequest(BaseModel):
    org_id: Optional[str] = Field(default=None, description="Optional target org_id filter")
    include_moderation: bool = Field(default=True, description="Include approved moderation queue items")
    include_arena_findings: bool = Field(default=True, description="Include completed Arena TestRun.findings")
    retrain: bool = Field(default=False, description="Trigger per-client train_client_models after merge")
    retrain_orgs: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit org_ids to retrain when retrain=True",
    )


@router.post("/submit", response_model=Dict[str, Any])
async def submit_attack(
    req: SubmitAttackRequest,
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """Submit an adversarial attack sample for moderation and poisoning inspection."""
    try:
        item = await queue.submit(
            attack_text=req.attack_text,
            category=req.category,
            severity=req.severity,
            source=req.source,
            domain=req.domain,
            metadata=req.metadata,
            bypass_moderation=req.bypass_moderation,
        )
        return item.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/items", response_model=List[Dict[str, Any]])
async def list_items(
    status: Optional[str] = Query(None, description="Filter by status (pending, approved, rejected)"),
    category: Optional[str] = Query(None, description="Filter by attack category"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> List[Dict[str, Any]]:
    """List items in the moderation queue with optional filtering."""
    status_enum = None
    if status:
        try:
            status_enum = ModerationStatus(status.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status filter '{status}'")

    items = await queue.list_items(
        status=status_enum,
        category=category,
        limit=limit,
        offset=offset,
    )
    return [i.to_dict() for i in items]


@router.get("/items/{item_id}", response_model=Dict[str, Any])
async def get_item(
    item_id: str,
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """Retrieve details of a single moderation item."""
    item = await queue.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Moderation item not found: {item_id}")
    return item.to_dict()


@router.post("/items/{item_id}/approve", response_model=Dict[str, Any])
async def approve_item(
    item_id: str,
    req: ReviewDecisionRequest,
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """Approve a pending adversarial sample and index it into Qdrant."""
    try:
        item = await queue.approve(
            item_id=item_id,
            reviewer_id=req.reviewer_id,
            reviewer_notes=req.notes,
        )
        return item.to_dict()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Moderation item not found: {item_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/items/{item_id}/reject", response_model=Dict[str, Any])
async def reject_item(
    item_id: str,
    req: RejectionRequest,
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """Reject an adversarial sample with rationale."""
    try:
        item = await queue.reject(
            item_id=item_id,
            reviewer_id=req.reviewer_id,
            reason=req.reason,
        )
        return item.to_dict()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Moderation item not found: {item_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/stats", response_model=Dict[str, Any])
async def get_queue_stats(
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """Retrieve moderation queue statistics."""
    return await queue.get_stats()


@router.post("/intel/ingest", response_model=Dict[str, Any])
async def ingest_threat_intel(
    req: IngestFeedRequest,
    ingester: ThreatIntelIngester = Depends(get_threat_ingester),
) -> Dict[str, Any]:
    """Ingest external threat feed and queue prompts for moderation."""
    try:
        result = await ingester.ingest_feed(
            feed_type=req.feed_type,
            data=req.data,
            feed_name=req.feed_name,
        )
        return result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/export-training", response_model=Dict[str, Any])
async def export_training_feedback(
    req: ExportTrainingRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    queue: AttackModerationQueue = Depends(get_moderation_queue),
) -> Dict[str, Any]:
    """
    Merge approved moderation items and completed Arena TestRun.findings into
    `backend/data/redteam_findings.json` and optionally trigger per-client retraining.
    """
    try:
        validated_org = validate_org_id(req.org_id) if (req.org_id is not None and req.org_id != "default") else req.org_id
        validated_retrain_orgs = (
            [validate_org_id(oid) for oid in req.retrain_orgs]
            if req.retrain_orgs
            else req.retrain_orgs
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scoped_org = enforce_tenant_access(auth_ctx, validated_org)
    target_retrain_orgs = validated_retrain_orgs
    if target_retrain_orgs:
        for oid in target_retrain_orgs:
            enforce_tenant_access(auth_ctx, oid)

    approved_items = (
        await queue.list_items(status=ModerationStatus.APPROVED, limit=1000)
        if req.include_moderation
        else []
    )
    queue_session_factory = queue._get_session_factory() if hasattr(queue, "_get_session_factory") else None
    queue_uses_db = bool(getattr(queue, "_use_db", True))
    try:
        return await asyncio.to_thread(
            export_feedback_to_training,
            org_id=scoped_org,
            moderation_items=approved_items,
            session_factory=queue_session_factory,
            use_db_session=queue_uses_db or req.include_arena_findings,
            include_moderation=req.include_moderation,
            include_arena_findings=req.include_arena_findings,
            retrain=req.retrain,
            retrain_orgs=target_retrain_orgs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed exporting moderation/Arena feedback to training data: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training export failed: {exc}") from exc
