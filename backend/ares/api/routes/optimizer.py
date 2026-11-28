"""
FastAPI REST Routes for ARES Prompt Optimizer Engine.
Provides endpoints to optimize prompts, retrieve intent categories, and inspect token savings.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ares.api.dependencies import enforce_tenant_access, verify_api_key
from ares.config import settings
from ares.intent.taxonomy import INTENT_DIRECTIVES, PromptIntent
from ares.optimizer.pipeline import (
    CompressionEngineKind,
    OptimizationStatus,
    OptimizedPromptResult,
    UserPromptOptimizer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["optimizer"])

# Shared optimizer instance
_optimizer: Optional[UserPromptOptimizer] = None


def get_prompt_optimizer() -> UserPromptOptimizer:
    """Dependency provider for UserPromptOptimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = UserPromptOptimizer(settings=settings)
    return _optimizer


class OptimizePromptRequest(BaseModel):
    """Request payload to optimize a user prompt."""
    prompt: str = Field(..., min_length=1, max_length=50000, description="The raw prompt text to optimize")
    domain: Optional[str] = Field(default=None, description="Optional domain hint (e.g. software, fintech, healthcare)")
    target_fidelity: Optional[float] = Field(default=None, ge=0.5, le=1.0, description="Minimum acceptable semantic similarity threshold (defaults to class/client-aware threshold)")
    allow_harmful: bool = Field(default=False, description="Whether to bypass security gate (admin + approved research policy only)")
    org_id: Optional[str] = Field(default=None, description="Optional organization/client ID for per-client classifier and fidelity thresholds")
    prefer_rule_compression: bool = Field(default=False, description="Prefer ultra-fast Stage-1 lexical compression when target reduction is met")
    inference_mode: str = Field(default="auto", description="Inference mode: auto, rules_adapter, client_adapter, lora, ollama, cloud")
    model_name: Optional[str] = Field(default=None, description="Optional explicit model identifier override")


class OptimizePromptResponse(BaseModel):
    """Response payload containing optimized prompt and analytics."""
    status: str
    original_prompt: str
    optimized_prompt: Optional[str]
    intent: str
    intent_confidence: float
    directive_name: str
    original_tokens: int
    optimized_tokens: int
    token_savings_pct: float
    token_savings_display: str
    semantic_fidelity: float
    target_fidelity: float = 0.85
    verification_passed: bool = True
    verification_status: str = "verified"
    mitigated_threats: List[str] = Field(default_factory=list)
    security_verdict: str
    security_risk_score: float
    latency_ms: float
    exemplars_used: int
    reasoning: Optional[str]
    org_id: str = "default"
    classifier_used: str = "global-sklearn"
    compression_engine: str = "rules_adapter:default"
    engine_kind: str = CompressionEngineKind.RULES_ADAPTER.value
    client_adapter_applied: bool = False
    client_lora_applied: bool = False
    protected_terms_preserved: bool = True
    missing_protected_terms: List[str] = Field(default_factory=list)


def _enforce_allow_harmful_policy(
    req: OptimizePromptRequest,
    auth_ctx: Dict[str, Any],
    research_policy_header: Optional[str],
) -> None:
    """Gate `allow_harmful` behind admin role + server override policy + explicit header."""
    if not req.allow_harmful:
        return
    if not auth_ctx.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="allow_harmful is restricted to authorized admin/research sessions and cannot be enabled by tenant keys.",
        )
    if (
        not settings.allow_optimizer_harmful_override
        or (research_policy_header or "").strip().lower() != "approved"
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "allow_harmful requires ARES_ALLOW_OPTIMIZER_HARMFUL_OVERRIDE=true "
                "and header 'X-Ares-Research-Policy: approved'."
            ),
        )


class IntentInfoResponse(BaseModel):
    """Metadata describing an intent class."""
    intent: str
    display_name: str
    target_token_reduction_pct: float
    preservation_rules: List[str]
    pruning_rules: List[str]
    structural_enforcements: List[str]


@router.post("/api/v1/optimizer/optimize", response_model=OptimizePromptResponse)
@router.post("/api/optimizer/optimize", response_model=OptimizePromptResponse)
async def optimize_prompt_endpoint(
    req: OptimizePromptRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    optimizer: UserPromptOptimizer = Depends(get_prompt_optimizer),
    x_ares_research_policy: Optional[str] = Header(default=None, alias="X-Ares-Research-Policy"),
) -> OptimizePromptResponse:
    """
    Analyze and optimize an unoptimized user prompt.
    Applies security gate, intent categorization, few-shot RAG retrieval,
    LLM/rules compression, and fail-closed semantic fidelity verification.
    """
    _enforce_allow_harmful_policy(req, auth_ctx, x_ares_research_policy)
    scoped_org_id = enforce_tenant_access(auth_ctx, req.org_id)
    try:
        res = await optimizer.optimize_prompt(
            user_prompt=req.prompt,
            domain=req.domain,
            target_fidelity=req.target_fidelity,
            allow_harmful=req.allow_harmful,
            org_id=scoped_org_id,
            prefer_rule_compression=req.prefer_rule_compression,
            inference_mode=req.inference_mode,
            model_name=req.model_name,
        )

        savings_display = f"{int(res.token_savings_pct * 100)}%"
        return OptimizePromptResponse(
            status=res.status.value,
            original_prompt=res.original_prompt,
            optimized_prompt=res.optimized_prompt,
            intent=res.intent.value,
            intent_confidence=res.intent_confidence,
            directive_name=res.directive_name,
            original_tokens=res.original_tokens,
            optimized_tokens=res.optimized_tokens,
            token_savings_pct=res.token_savings_pct,
            token_savings_display=savings_display,
            semantic_fidelity=res.semantic_fidelity,
            target_fidelity=res.target_fidelity,
            verification_passed=res.verification_passed,
            verification_status=res.verification_status,
            mitigated_threats=res.mitigated_threats,
            security_verdict=res.security_verdict,
            security_risk_score=res.security_risk_score,
            latency_ms=res.latency_ms,
            exemplars_used=res.exemplars_used,
            reasoning=res.reasoning,
            org_id=res.org_id,
            classifier_used=res.classifier_used,
            compression_engine=res.compression_engine,
            engine_kind=res.engine_kind,
            client_adapter_applied=res.client_adapter_applied,
            client_lora_applied=res.client_lora_applied,
            protected_terms_preserved=res.protected_terms_preserved,
            missing_protected_terms=res.missing_protected_terms,
        )
    except Exception as exc:
        logger.error("Error optimizing prompt: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt optimization failed: {str(exc)}",
        ) from exc


@router.post("/api/v1/optimizer/optimize/stream")
@router.post("/api/optimizer/optimize/stream")
async def optimize_prompt_stream_endpoint(
    req: OptimizePromptRequest,
    auth_ctx: Dict[str, Any] = Depends(verify_api_key),
    optimizer: UserPromptOptimizer = Depends(get_prompt_optimizer),
    x_ares_research_policy: Optional[str] = Header(default=None, alias="X-Ares-Research-Policy"),
) -> StreamingResponse:
    """
    Stream real-time Server-Sent Events (SSE) for each stage of the prompt optimization pipeline.
    """
    _enforce_allow_harmful_policy(req, auth_ctx, x_ares_research_policy)
    scoped_org_id = enforce_tenant_access(auth_ctx, req.org_id)

    async def _event_generator():
        try:
            async for event in optimizer.optimize_prompt_stream(
                user_prompt=req.prompt,
                domain=req.domain,
                target_fidelity=req.target_fidelity,
                allow_harmful=req.allow_harmful,
                org_id=scoped_org_id,
                prefer_rule_compression=req.prefer_rule_compression,
                inference_mode=req.inference_mode,
                model_name=req.model_name,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("Error in optimization stream: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@router.get("/api/v1/optimizer/intents", response_model=List[IntentInfoResponse])
@router.get("/api/optimizer/intents", response_model=List[IntentInfoResponse])
async def get_intent_categories() -> List[IntentInfoResponse]:
    """Return all supported intent categories and their optimization directives."""
    results: List[IntentInfoResponse] = []
    for intent, directive in INTENT_DIRECTIVES.items():
        results.append(
            IntentInfoResponse(
                intent=intent.value,
                display_name=directive.display_name,
                target_token_reduction_pct=directive.target_token_reduction_pct,
                preservation_rules=directive.preservation_rules,
                pruning_rules=directive.pruning_rules,
                structural_enforcements=directive.structural_enforcements,
            )
        )
    return results
