"""
FastAPI REST Routes for ARES Prompt Optimizer Engine.
Provides endpoints to optimize prompts, retrieve intent categories, and inspect token savings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ares.config import settings
from ares.intent.taxonomy import INTENT_DIRECTIVES, PromptIntent
from ares.optimizer.pipeline import (
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
    target_fidelity: Optional[float] = Field(default=None, ge=0.5, le=1.0, description="Minimum acceptable semantic similarity threshold (defaults to class-aware threshold)")
    allow_harmful: bool = Field(default=False, description="Whether to bypass security gate (for red-teaming research)")


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
    mitigated_threats: List[str] = Field(default_factory=list)
    security_verdict: str
    security_risk_score: float
    latency_ms: float
    exemplars_used: int
    reasoning: Optional[str]


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
    optimizer: UserPromptOptimizer = Depends(get_prompt_optimizer),
) -> OptimizePromptResponse:
    """
    Analyze and optimize an unoptimized user prompt.
    Applies security gate, intent categorization, few-shot RAG retrieval,
    Llama 3.1 restructuring, and semantic fidelity checks.
    """
    try:
        res = await optimizer.optimize_prompt(
            user_prompt=req.prompt,
            domain=req.domain,
            target_fidelity=req.target_fidelity,
            allow_harmful=req.allow_harmful,
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
            mitigated_threats=res.mitigated_threats,
            security_verdict=res.security_verdict,
            security_risk_score=res.security_risk_score,
            latency_ms=res.latency_ms,
            exemplars_used=res.exemplars_used,
            reasoning=res.reasoning,
        )
    except Exception as exc:
        logger.error("Error optimizing prompt: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt optimization failed: {str(exc)}",
        ) from exc


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
