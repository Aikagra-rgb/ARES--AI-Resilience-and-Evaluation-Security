"""
Batch Optimization Endpoint for ARES Prompt Optimizer
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio

from ares.config import settings
from ares.optimizer.pipeline import UserPromptOptimizer

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

class BatchOptimizeItem(BaseModel):
    id: Optional[str] = None
    prompt: str = Field(..., description="Raw prompt to optimize")
    domain: Optional[str] = Field(default=None, description="Optional domain context")
    target_fidelity: float = Field(default=0.85, ge=0.0, le=1.0)
    allow_harmful: bool = Field(default=False)

class BatchOptimizeRequest(BaseModel):
    requests: List[BatchOptimizeItem]
    concurrency: int = Field(default=4, ge=1, le=16)

class BatchItemResult(BaseModel):
    id: Optional[str] = None
    original_prompt: str
    optimized_prompt: Optional[str]
    original_tokens: int
    optimized_tokens: int
    token_savings_pct: float
    semantic_fidelity: float
    status: str
    error: Optional[str] = None

class BatchOptimizeResponse(BaseModel):
    total_processed: int
    total_original_tokens: int
    total_optimized_tokens: int
    overall_savings_pct: float
    results: List[BatchItemResult]

@router.post("/optimize", response_model=BatchOptimizeResponse)
async def batch_optimize(req: BatchOptimizeRequest):
    """
    Optimize multiple enterprise prompts concurrently.
    """
    optimizer = UserPromptOptimizer(settings=settings)
    semaphore = asyncio.Semaphore(req.concurrency)

    async def _process_item(item: BatchOptimizeItem) -> BatchItemResult:
        async with semaphore:
            try:
                res = await optimizer.optimize_prompt(
                    user_prompt=item.prompt,
                    domain=item.domain,
                    target_fidelity=item.target_fidelity,
                    allow_harmful=item.allow_harmful,
                )
                return BatchItemResult(
                    id=item.id,
                    original_prompt=res.original_prompt,
                    optimized_prompt=res.optimized_prompt,
                    original_tokens=res.original_tokens,
                    optimized_tokens=res.optimized_tokens,
                    token_savings_pct=res.token_savings_pct,
                    semantic_fidelity=res.semantic_fidelity,
                    status=res.status.value if hasattr(res.status, "value") else str(res.status),
                )
            except Exception as exc:
                return BatchItemResult(
                    id=item.id,
                    original_prompt=item.prompt,
                    optimized_prompt=item.prompt,
                    original_tokens=len(item.prompt.split()),
                    optimized_tokens=len(item.prompt.split()),
                    token_savings_pct=0.0,
                    semantic_fidelity=1.0,
                    status="error",
                    error=str(exc),
                )

    tasks = [_process_item(item) for item in req.requests]
    results = await asyncio.gather(*tasks)

    orig_total = sum(r.original_tokens for r in results)
    opt_total = sum(r.optimized_tokens for r in results)
    overall_savings = (orig_total - opt_total) / orig_total if orig_total > 0 else 0.0

    return BatchOptimizeResponse(
        total_processed=len(results),
        total_original_tokens=orig_total,
        total_optimized_tokens=opt_total,
        overall_savings_pct=round(overall_savings, 4),
        results=results,
    )
