"""
Taxonomy and directives for ARES Intent-Driven Prompt Optimization.
Defines the core prompt intent classes, optimization strategies, and sample schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptIntent(str, Enum):
    """Core intent categories for enterprise and developer prompts."""
    CODING_TECHNICAL = "coding_technical"
    DATA_EXTRACTION_JSON = "data_extraction_json"
    ANALYTICAL_QA = "analytical_qa"
    CREATIVE_WRITING = "creative_writing"
    SYSTEM_DIRECTIVE = "system_directive"


class OptimizationDirective(BaseModel):
    """
    Rules and configuration that dictate how the generative LLM (Llama 3.1)
    optimizes prompts belonging to a specific intent class.
    """
    intent: PromptIntent
    display_name: str
    target_token_reduction_pct: float = Field(
        description="Target token reduction percentage (e.g. 0.35 for 35%)"
    )
    preservation_rules: List[str] = Field(
        description="Aspects of the prompt that must never be altered or pruned"
    )
    pruning_rules: List[str] = Field(
        description="Elements that should be aggressively stripped or compressed"
    )
    structural_enforcements: List[str] = Field(
        description="Formatting structures to inject (e.g., XML tags, Markdown fences)"
    )
    system_instruction: str = Field(
        description="Specialized prompt guidance given to the optimizer LLM"
    )


class IntentSample(BaseModel):
    """A single training, validation, or test prompt sample."""
    id: str
    prompt: str
    intent: PromptIntent
    domain: str = "general"
    verbose_ratio: float = 1.0
    source: str = "synthetic"  # synthetic, seed, public_hf, redteam
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Default Directives for the 5 Intent Classes
INTENT_DIRECTIVES: Dict[PromptIntent, OptimizationDirective] = {
    PromptIntent.CODING_TECHNICAL: OptimizationDirective(
        intent=PromptIntent.CODING_TECHNICAL,
        display_name="Coding & Technical Engineering",
        target_token_reduction_pct=0.30,
        preservation_rules=[
            "Preserve all programming languages, code snippets, syntax, and libraries verbatim.",
            "Retain exact variable, function, class, and parameter names.",
            "Keep algorithmic constraints, time/space complexity requirements, and edge-case specs.",
        ],
        pruning_rules=[
            "Remove conversational pleasantries ('Hey could you please write me a script that...')",
            "Eliminate redundant descriptive commentary and duplicate requirement statements.",
        ],
        structural_enforcements=[
            "Use clear markdown code fences.",
            "Specify input/output types and error-handling requirements declaratively.",
        ],
        system_instruction=(
            "Focus on absolute technical precision. Strip all conversational conversational filler. "
            "Never alter or abbreviate code blocks, library imports, function signatures, or variable names. "
            "Formulate crisp, declarative technical requirements with explicit I/O constraints."
        ),
    ),
    PromptIntent.DATA_EXTRACTION_JSON: OptimizationDirective(
        intent=PromptIntent.DATA_EXTRACTION_JSON,
        display_name="Data Extraction & Structured Output",
        target_token_reduction_pct=0.45,
        preservation_rules=[
            "Retain all schema field names, data types, and required key constraints.",
            "Preserve exact source text or documents targeted for extraction.",
        ],
        pruning_rules=[
            "Aggressively prune natural language explanations of how extraction works.",
            "Eliminate conversational intros and unneeded preambles.",
        ],
        structural_enforcements=[
            "Enforce strict JSON schema template specifications.",
            "Add 'Return ONLY valid JSON without markdown fences' constraint.",
        ],
        system_instruction=(
            "Transform wordy extraction instructions into a high-density, schema-driven prompt. "
            "Replace long-winded descriptive text with a concise JSON template schema. "
            "Explicitly mandate raw JSON output with zero explanatory preface."
        ),
    ),
    PromptIntent.ANALYTICAL_QA: OptimizationDirective(
        intent=PromptIntent.ANALYTICAL_QA,
        display_name="Analytical Q&A & Research",
        target_token_reduction_pct=0.40,
        preservation_rules=[
            "Preserve core analytical question, context entities, and factual scope.",
            "Retain required analytical dimensions (e.g. pros/cons, metrics, timeframes).",
        ],
        pruning_rules=[
            "Prune rambling background setup, apologetic phrases, and redundant restatements.",
            "Condense verbose contextual descriptions into bulleted premise tokens.",
        ],
        structural_enforcements=[
            "Use structured bullet points for multi-part inquiries.",
            "Define expected answer format (e.g. summary table, bulleted takeaways).",
        ],
        system_instruction=(
            "Synthesize the user's inquiry into an authoritative, dense research prompt. "
            "Eliminate stream-of-consciousness filler. Structure compound questions as crisp bullet points. "
            "Maximize token compression while safeguarding factual context and query nuance."
        ),
    ),
    PromptIntent.CREATIVE_WRITING: OptimizationDirective(
        intent=PromptIntent.CREATIVE_WRITING,
        display_name="Creative Drafting & Narrative",
        target_token_reduction_pct=0.20,
        preservation_rules=[
            "Preserve tone of voice, stylistic adjectives, emotional nuance, and persona.",
            "Keep plot points, brand voice guidelines, and stylistic constraints.",
        ],
        pruning_rules=[
            "Remove meta-talk about writing ('I am writing a story and I need ideas for...')",
            "Eliminate redundant synonyms and repetitive instructions.",
        ],
        structural_enforcements=[
            "Organize into Persona, Context, Audience, and Format sections.",
        ],
        system_instruction=(
            "Carefully preserve creative voice, tone, and stylistic atmosphere. "
            "Do NOT over-compress adjectives or stylistic requests. "
            "Only strip meta-conversational filler and clarify structural formatting."
        ),
    ),
    PromptIntent.SYSTEM_DIRECTIVE: OptimizationDirective(
        intent=PromptIntent.SYSTEM_DIRECTIVE,
        display_name="System Instructions & Agent Personas",
        target_token_reduction_pct=0.35,
        preservation_rules=[
            "Preserve all business logic, permission boundaries, and capability rules.",
            "Keep safety constraints and confidential data protection policies.",
        ],
        pruning_rules=[
            "Eliminate polite or soft-spoken language ('Please try to be helpful and do your best...')",
            "Deduplicate overlapping behavioral constraints.",
        ],
        structural_enforcements=[
            "Enforce strict XML delimiter boundaries (`<rules>`, `<user_input>`).",
            "Inject adversarial immunity constraints against prompt injection.",
        ],
        system_instruction=(
            "Harden and compress the system instruction. Use dense imperative commands. "
            "Isolate untrusted user data with strict boundary tags `<user_input>`. "
            "Eliminate duplicate rules and establish a resilient zero-trust security perimeter."
        ),
    ),
}


def get_directive_for_intent(intent: PromptIntent) -> OptimizationDirective:
    """Retrieve the optimization directive associated with an intent class."""
    return INTENT_DIRECTIVES.get(intent, INTENT_DIRECTIVES[PromptIntent.ANALYTICAL_QA])
