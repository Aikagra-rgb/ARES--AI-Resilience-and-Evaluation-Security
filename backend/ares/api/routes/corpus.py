"""Corpus endpoint — embed and save verified attack breaches to Qdrant."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException, status
from qdrant_client import models

from ares.api.schemas import CorpusSaveRequest, CorpusSaveResponse
from ares.config import settings
from ares.mappings import to_backend_category
from ares.vectordb.embedder import GeminiEmbedder
from ares.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["corpus"])


@router.post("/corpus", response_model=CorpusSaveResponse)
async def save_to_corpus(request: CorpusSaveRequest) -> CorpusSaveResponse:
    """
    Permanently save a verified attack payload to the Qdrant corpus.

    Safety:
    To prevent corpus poisoning, submissions must be anchored to an evaluation test run
    or verified analyst finding with positive breach evidence, or supply an explicit
    verified exploit payload with a mandatory analyst rationale.
    """
    from ares.api.routes.tests import _test_store, _load_test_run_from_db

    corr_id = uuid.uuid4().hex[:12]
    point_id = str(uuid.uuid4())
    test_id = request.test_id or f"test-manual-{uuid.uuid4().hex[:8]}"

    # 1. Resolve attack text, category and verification evidence
    test_run = _test_store.get(request.test_id) if request.test_id else None
    if not test_run and request.test_id:
        test_run = _load_test_run_from_db(request.test_id)

    attack_text: str = ""
    category_val: str = "instruction_override"
    domain: str = "general"
    severity: str = request.severity or "high"
    is_verified_breach: bool = False

    if request.attack_text and request.attack_text.strip():
        attack_text = request.attack_text.strip()
        if len(attack_text) < 5:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Attack text is too short to constitute a valid payload.",
            )
        category_val = to_backend_category(request.category or "instruction_override").value
        domain = "analyst_submission"
        # Require an analyst note or verified flag for unanchored manual submissions
        if not request.analyst_note or len(request.analyst_note.strip()) < 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manual corpus additions require an analyst_note documenting the verified breach.",
            )
        is_verified_breach = True
    elif test_run:
        cfg = test_run.configuration
        category_val = to_backend_category(
            cfg.attack_categories[0] if cfg.attack_categories else "instruction_override"
        ).value
        domain = "adversarial_eval"
        
        # Check if the evaluation actually confirmed a breach
        if test_run.analysis and test_run.analysis.attack_succeeded:
            is_verified_breach = True
            severity = str(test_run.analysis.severity.value if hasattr(test_run.analysis.severity, "value") else test_run.analysis.severity).lower()
            attack_text = cfg.user_prompt or cfg.system_prompt or f"Verified breach against {cfg.target_application}"
        else:
            # If the test did not breach, cannot store as a verified attack unless explicit verified payload was given
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Test run {request.test_id} did not breach the target model. Clean runs cannot be saved to the attack corpus.",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either attack_text with an analyst_note or a valid test_id with verified breaches is required.",
        )

    # 2. Embed payload vector
    embedder = GeminiEmbedder(settings=settings)
    store = QdrantStore(settings=settings)
    store.ensure_collection(store.collection_name)

    vectors = await embedder.embed_batch([attack_text])
    vector = vectors[0] if vectors else [0.0] * settings.qdrant_vector_size

    # 3. Upsert to Qdrant with standardized schema
    payload = {
        "attack_text": attack_text,
        "category": category_val,
        "source": "analyst_corpus_save" if request.attack_text else "verified_test_evaluation",
        "severity": severity,
        "tags": ["analyst_saved", test_id],
        "domain": domain,
        "success": is_verified_breach,
        "rule_verdict": is_verified_breach,
        "llm_verdict": is_verified_breach,
        "reasoning": request.analyst_note or f"Verified attack breach from test run {test_id}",
        "canary_tokens_found": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    store.client.upsert(
        collection_name=store.collection_name,
        points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
    )

    return CorpusSaveResponse(
        attack_id=point_id,
        test_id=test_id,
        saved_at=datetime.now(timezone.utc),
        correlation_id=corr_id,
    )
