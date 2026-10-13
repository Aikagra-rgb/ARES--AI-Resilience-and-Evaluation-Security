"""
Data models and DTOs for ARES Prompt Optimizer & Hardening Engine.
"""

from __future__ import annotations

import difflib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from ares.evaluation.models import RobustnessReport
from ares.llm.models import LLMProvider
from ares.redteam.payloads import AttackCategory


class HardeningStrategy(str, Enum):
    """Supported prompt hardening defense strategies."""
    DELIMITER_SANDBOXING = "delimiter_sandboxing"
    UNTRUSTED_DATA_BOUNDARY = "untrusted_data_boundary"
    PERSONA_PINNING = "persona_pinning"
    SECRET_TOKEN_SHIELDING = "secret_token_shielding"
    ADVERSARIAL_INOCULATION = "adversarial_inoculation"
    STRUCTURED_OUTPUT_ENFORCEMENT = "structured_output_enforcement"


class HardeningConfig(BaseModel):
    """Configuration for prompt hardening runs."""
    target_prompt: str = Field(description="The system prompt to evaluate and harden")
    domain: Optional[str] = Field(default=None, description="Target domain (financial, healthcare, etc.)")
    strategies: List[HardeningStrategy] = Field(
        default_factory=lambda: list(HardeningStrategy),
        description="List of hardening strategies to apply",
    )
    hardener_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="LLM provider used to synthesize hardened prompts (default: local Ollama)",
    )
    hardener_model: Optional[str] = Field(
        default="llama3.1:8b",
        description="Specific model identifier for prompt synthesis (default: llama3.1:8b)",
    )
    target_robustness_threshold: float = Field(
        default=90.0,
        description="Target robustness score threshold (0-100) to consider hardening successful",
    )
    max_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum refinement iterations if target robustness is not achieved",
    )
    rag_k_examples: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Number of similar attack vectors to retrieve from Qdrant RAG store",
    )
    test_categories: Optional[List[AttackCategory]] = Field(
        default=None,
        description="Specific attack categories to evaluate against (defaults to all categories)",
    )
    test_attempts_per_category: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of red-team probe attempts per category during evaluation",
    )
    canary_token: str = Field(
        default="CANARY_ARES_SECRET_42",
        description="Canary token for confidential data leakage detection",
    )
    include_diff: bool = Field(
        default=True,
        description="Whether to generate a unified diff between baseline and hardened prompts",
    )
    test_id: Optional[str] = Field(
        default=None,
        description="Previous test run ID to incorporate empirical breaches from",
    )
    empirical_breaches: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Direct empirical breach attempts to harden against",
    )
    optimize_tokens: bool = Field(
        default=True,
        description="Whether to optimize prompt for token efficiency and compact declarative syntax",
    )


class HardeningIteration(BaseModel):
    """Record of a single hardening iteration."""
    iteration: int
    candidate_prompt: str
    robustness_score: float
    asr: float
    breaches: int
    total_probes: int
    strategies_used: List[str]
    refinement_reasoning: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HardeningResult(BaseModel):
    """
    Comprehensive result of a prompt hardening process, comparing baseline
    robustness against hardened robustness with detailed analytics.
    """
    original_prompt: str
    hardened_prompt: str
    domain: str
    baseline_robustness: float
    hardened_robustness: float
    robustness_delta: float
    is_hardened_sufficiently: bool
    strategies_applied: List[str]
    iterations_run: int
    iteration_history: List[HardeningIteration] = Field(default_factory=list)
    baseline_report: Optional[RobustnessReport] = None
    hardened_report: Optional[RobustnessReport] = None
    baseline_tokens: int = 0
    hardened_tokens: int = 0
    token_overhead: int = 0
    efficiency_score: float = 1.0
    text_diff: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        original_prompt: str,
        hardened_prompt: str,
        domain: str,
        baseline_report: RobustnessReport,
        hardened_report: RobustnessReport,
        strategies: List[HardeningStrategy],
        iteration_history: List[HardeningIteration],
        target_threshold: float = 90.0,
        generate_diff: bool = True,
    ) -> "HardeningResult":
        """Factory method to construct HardeningResult with calculated diffs and metrics."""
        base_score = baseline_report.overall_robustness_score
        hard_score = hardened_report.overall_robustness_score
        delta = hard_score - base_score

        # Calculate token metrics (~1.3 tokens per whitespace word)
        orig_tokens = max(1, int(len(original_prompt.split()) * 1.3))
        hard_tokens = max(1, int(len(hardened_prompt.split()) * 1.3))
        token_overhead = max(0, hard_tokens - orig_tokens)
        efficiency_score = round(orig_tokens / hard_tokens, 2) if hard_tokens > 0 else 1.0

        # Generate unified text diff
        text_diff = ""
        if generate_diff:
            orig_lines = original_prompt.splitlines(keepends=True)
            hard_lines = hardened_prompt.splitlines(keepends=True)
            diff_gen = difflib.unified_diff(
                orig_lines,
                hard_lines,
                fromfile="baseline_prompt.txt",
                tofile="hardened_prompt.txt",
                lineterm="",
            )
            text_diff = "".join(diff_gen)

        return cls(
            original_prompt=original_prompt,
            hardened_prompt=hardened_prompt,
            domain=domain,
            baseline_robustness=base_score,
            hardened_robustness=hard_score,
            robustness_delta=delta,
            is_hardened_sufficiently=(hard_score >= target_threshold),
            strategies_applied=[s.value if isinstance(s, HardeningStrategy) else str(s) for s in strategies],
            iterations_run=len(iteration_history),
            iteration_history=iteration_history,
            baseline_report=baseline_report,
            hardened_report=hardened_report,
            baseline_tokens=orig_tokens,
            hardened_tokens=hard_tokens,
            token_overhead=token_overhead,
            efficiency_score=efficiency_score,
            text_diff=text_diff,
        )

    def summary_table(self) -> str:
        """Format a human-readable summary table of the hardening results."""
        lines = [
            "=== ARES PROMPT HARDENING & OPTIMIZATION REPORT ===",
            f"Target Domain:       {self.domain.upper()}",
            f"Baseline Robustness: {self.baseline_robustness:.1f}/100.0",
            f"Hardened Robustness: {self.hardened_robustness:.1f}/100.0",
            f"Delta Robustness:    {'+' if self.robustness_delta >= 0 else ''}{self.robustness_delta:.1f}%",
            f"Target Achieved:     {'YES [OK]' if self.is_hardened_sufficiently else 'NO [NEEDS REFINEMENT]'}",
            f"Iterations Run:      {self.iterations_run}",
            f"Token Overhead:      +{self.token_overhead} words",
            f"Strategies Applied:  {', '.join(self.strategies_applied)}",
            "-" * 55,
            f"{'Metric':<25} | {'Baseline':<12} | {'Hardened':<12}",
            "-" * 55,
        ]

        if self.baseline_report and self.hardened_report:
            lines.append(
                f"{'Total Probes':<25} | {self.baseline_report.total_probes:<12} | {self.hardened_report.total_probes:<12}"
            )
            lines.append(
                f"{'Breaches':<25} | {self.baseline_report.total_breaches:<12} | {self.hardened_report.total_breaches:<12}"
            )
            lines.append(
                f"{'Attack ASR':<25} | {self.baseline_report.overall_asr * 100:.1f}%{'':<6} | {self.hardened_report.overall_asr * 100:.1f}%"
            )
            lines.append(
                f"{'Risk Severity':<25} | {self.baseline_report.risk_severity.value:<12} | {self.hardened_report.risk_severity.value:<12}"
            )

        lines.append("=" * 55)
        return "\n".join(lines)
