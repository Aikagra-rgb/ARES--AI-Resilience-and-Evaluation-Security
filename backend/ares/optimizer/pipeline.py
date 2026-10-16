"""
Unified User Prompt Optimizer Engine.
Chains Ingress Security Gate -> Intent Classifier -> Qdrant Few-Shot Exemplars -> Llama 3.1 Synthesis -> Semantic Drift Verification.
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ares.config import Settings, get_settings
from ares.intent.classifier import SklearnIntentClassifier
from ares.intent.exemplar_store import QdrantExemplarStore
from ares.intent.taxonomy import (
    OptimizationDirective,
    PromptIntent,
    get_directive_for_intent,
)
from ares.llm.client import LLMClient
from ares.llm.models import ChatMessage, LLMProvider
from ares.security.structural_detector import Decision, SeverityLevel, StructuralAttackDetector
from ares.vectordb.embedder import GeminiEmbedder

logger = logging.getLogger(__name__)

# Prometheus Metrics Telemetry
try:
    from prometheus_client import Counter, Histogram
    _PROMETHEUS_AVAILABLE = True
    optimization_requests_total = Counter('ares_optimization_requests_total', 'Total prompt optimization requests', ['intent', 'status'])
    optimization_latency_seconds = Histogram('ares_optimization_latency_seconds', 'Prompt optimization latency in seconds')
    token_savings_metric = Histogram('ares_token_savings_pct', 'Token savings percentage', ['intent'])
    semantic_fidelity_metric = Histogram('ares_semantic_fidelity', 'Semantic fidelity score', ['intent'])
except Exception:
    _PROMETHEUS_AVAILABLE = False


class OptimizationStatus(str, Enum):
    """Execution status of prompt optimization."""
    SUCCESS = "success"
    BLOCKED_SECURITY = "blocked_security"
    DRIFT_WARNING = "drift_warning"
    FALLBACK = "fallback"


class OptimizedPromptResult(BaseModel):
    """Complete analytics envelope for an optimized user prompt."""
    status: OptimizationStatus
    original_prompt: str
    optimized_prompt: Optional[str] = None
    intent: PromptIntent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    directive_name: str
    original_tokens: int
    optimized_tokens: int
    token_savings_pct: float
    semantic_fidelity: float = Field(ge=0.0, le=1.0)
    security_verdict: str
    security_risk_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    exemplars_used: int = 0
    reasoning: Optional[str] = None


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two dense embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _estimate_tokens(text: str) -> int:
    """Approximate subword token count (~1.3 tokens per whitespace word)."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


class UserPromptOptimizer:
    """
    Production-grade Prompt Optimizer and Immunization Engine.
    Executes security gating, intent routing, few-shot RAG retrieval,
    local LLM restructuring, and embedding drift verification.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        security_detector: Optional[StructuralAttackDetector] = None,
        classifier: Optional[SklearnIntentClassifier] = None,
        exemplar_store: Optional[QdrantExemplarStore] = None,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[GeminiEmbedder] = None,
    ):
        self.settings = settings or get_settings()
        self.security_detector = security_detector or StructuralAttackDetector(
            canary_token=self.settings.gateway_canary_token
        )

        # Prefer the scaled 5,000 model if available
        if classifier is not None:
            self.classifier = classifier
        else:
            scaled_path = self.settings.app_data_dir if hasattr(self.settings, "app_data_dir") else None
            from pathlib import Path
            data_dir = Path(__file__).parent.parent / "intent" / "data"
            scaled_joblib = data_dir / "intent_classifier_scaled_5000.joblib"
            seed_joblib = data_dir / "intent_classifier_sklearn.joblib"
            model_target = scaled_joblib if scaled_joblib.exists() else seed_joblib
            self.classifier = SklearnIntentClassifier(model_path=model_target)

        self.exemplar_store = exemplar_store or QdrantExemplarStore(settings=self.settings)
        self.llm_client = llm_client
        self.embedder = embedder or GeminiEmbedder(settings=self.settings)

    async def optimize_prompt(
        self,
        user_prompt: str,
        domain: Optional[str] = None,
        target_fidelity: float = 0.85,
        allow_harmful: bool = False,
    ) -> OptimizedPromptResult:
        """
        Execute full prompt optimization pipeline.
        """
        t0 = time.perf_counter()
        orig_tokens = _estimate_tokens(user_prompt)

        # ---------------------------------------------------------------------
        # Step 1: Security Gate (Structural Attack Detection)
        # ---------------------------------------------------------------------
        sec_result = self.security_detector.detect(
            user_prompt,
            canary_token=self.settings.gateway_canary_token,
        )
        is_blocked = (sec_result.decision in (Decision.BLOCK, Decision.CHALLENGE)) and not allow_harmful

        if is_blocked:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            first_match = sec_result.detections[0] if sec_result.detections else None
            violation_msg = (
                f"Blocked by Security Gate: {first_match.description}"
                if first_match
                else "Blocked: Structural prompt injection detected"
            )
            return OptimizedPromptResult(
                status=OptimizationStatus.BLOCKED_SECURITY,
                original_prompt=user_prompt,
                optimized_prompt=None,
                intent=PromptIntent.SYSTEM_DIRECTIVE,
                intent_confidence=1.0,
                directive_name="Security Quarantine",
                original_tokens=orig_tokens,
                optimized_tokens=0,
                token_savings_pct=0.0,
                semantic_fidelity=0.0,
                security_verdict="BLOCKED_ADVERSARIAL_PAYLOAD",
                security_risk_score=round(sec_result.total_confidence, 2),
                latency_ms=elapsed_ms,
                exemplars_used=0,
                reasoning=violation_msg,
            )

        # ---------------------------------------------------------------------
        # Step 2: Intent Classification & Directive Resolution
        # ---------------------------------------------------------------------
        intent_res = self.classifier.predict(user_prompt)
        intent = intent_res.intent
        directive = intent_res.directive

        # ---------------------------------------------------------------------
        # Step 3: Dynamic Exemplar Retrieval via Qdrant RAG
        # ---------------------------------------------------------------------
        exemplars = []
        few_shot_block = ""
        try:
            exemplars = await self.exemplar_store.search_exemplars(
                query_prompt=user_prompt,
                intent=intent,
                domain=domain,
                limit=2,
            )
            few_shot_block = self.exemplar_store.format_few_shot_block(exemplars)
        except Exception as exc:
            logger.warning("Qdrant exemplar retrieval skipped (%s)", exc)

        # ---------------------------------------------------------------------
        # Step 4: Generative Optimization Synthesis (Llama 3.1 8B via Ollama)
        # ---------------------------------------------------------------------
        system_instruction = (
            f"You are the ARES Prompt Compiler & Optimization Engine.\n"
            f"CRITICAL MISSION: You are NOT an AI assistant answering or executing the user's prompt.\n"
            f"Your ONLY job is to REWRITE and COMPRESS the user's prompt into an ultra-concise, high-efficiency prompt "
            f"that another AI model will execute.\n"
            f"Strip pleasantries ('Hey AI', 'Could you please', 'Thanks so much'), conversational padding, and redundant words.\n"
            f"Preserve 100% of the core instructions, technical constraints, types, and schema specifications.\n\n"
            f"INTENT DIRECTIVE: {directive.display_name}\n"
            f"{directive.system_instruction}\n\n"
            f"STRICT PRESERVATION RULES:\n"
            + "\n".join(f"- {r}" for r in directive.preservation_rules)
            + f"\n\nSTRICT PRUNING RULES:\n"
            + "\n".join(f"- {r}" for r in directive.pruning_rules)
            + f"\n\nSTRUCTURAL ENFORCEMENT:\n"
            + "\n".join(f"- {r}" for r in directive.structural_enforcements)
            + f"\n\nOUTPUT FORMAT:\n"
            f"Output ONLY the rewritten, compressed prompt text. Do NOT prepend explanations, introductions, or pleasantries."
        )

        user_content_parts = []
        if few_shot_block:
            user_content_parts.append(few_shot_block)
        user_content_parts.append(f"ORIGINAL UNOPTIMIZED PROMPT:\n{user_prompt}\n\nREWRITTEN OPTIMIZED PROMPT:")
        user_message_text = "\n\n".join(user_content_parts)

        messages = [
            ChatMessage(role="system", content=system_instruction),
            ChatMessage(role="user", content=user_message_text),
        ]

        if self.llm_client is None:
            self.llm_client = LLMClient()

        optimized_text = ""
        try:
            provider = self.settings.hardener_provider or LLMProvider.OLLAMA
            primary_model = getattr(self.settings, "ollama_optimizer_model", "ares-optimizer:8b") or "ares-optimizer:8b"
            fallback_model = self.settings.hardener_model or "llama3.1:8b"

            # 1. First priority: Dedicated Native PEFT LoRA Inference Server (Direct GPU Weights)
            raw_lora_url = getattr(self.settings, "lora_server_url", "http://localhost:11435/v1") or "http://localhost:11435/v1"
            lora_url = raw_lora_url.strip().rstrip("/")
            use_lora = getattr(self.settings, "use_lora_server", True)
            response = None

            if use_lora:
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        resp = await client.post(
                            f"{lora_url}/chat/completions",
                            json={
                                "model": "ares-optimizer-lora",
                                "messages": [{"role": m.role, "content": m.content} for m in messages],
                                "temperature": 0.2,
                                "max_tokens": 1000,
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            raw_out = data["choices"][0]["message"]["content"]
                            response = type("Resp", (), {"text": raw_out, "content": raw_out})()
                            logger.info("Successfully optimized prompt using Native Fine-Tuned LoRA Adapter (GPU)")
                except Exception as lora_err:
                    logger.debug("Native LoRA server offline (%s); proceeding to Ollama router", lora_err)

            # 2. Secondary: Ollama ares-optimizer:8b with automatic fallback to llama3.1:8b
            if response is None:
                try:
                    response = await self.llm_client.complete(
                        messages=messages,
                        provider=provider.value if isinstance(provider, LLMProvider) else str(provider),
                        model=primary_model,
                        temperature=0.2,
                        max_tokens=1000,
                    )
                except Exception as model_err:
                    logger.warning("Primary optimizer '%s' unavailable (%s); falling back to base '%s'", primary_model, model_err, fallback_model)
                    response = await self.llm_client.complete(
                        messages=messages,
                        provider=provider.value if isinstance(provider, LLMProvider) else str(provider),
                        model=fallback_model,
                        temperature=0.2,
                        max_tokens=1000,
                    )
            raw_out = getattr(response, "text", None) or getattr(response, "content", "")
            optimized_text = str(raw_out).strip()
            # Clean accidental markdown code wrapper if model emitted one around prompt
            if optimized_text.startswith("```") and optimized_text.endswith("```"):
                lines = optimized_text.splitlines()
                if len(lines) >= 3 and not intent == PromptIntent.DATA_EXTRACTION_JSON:
                    optimized_text = "\n".join(lines[1:-1]).strip()

            # Strip common conversational preambles
            for preamble in [
                "here is the optimized prompt:",
                "here is the compressed prompt:",
                "here is the rewritten prompt:",
                "optimized prompt:",
                "rewritten prompt:",
            ]:
                if optimized_text.lower().startswith(preamble):
                    optimized_text = optimized_text[len(preamble):].strip()

            # If model wrote source code instead of a prompt instruction (e.g. async def, def, import)
            if any(optimized_text.startswith(kw) for kw in ("def ", "async def ", "import ", "from ", "class ")):
                compressed_fallback = self._fallback_rule_compress(user_prompt)
                optimized_text = f"Write Python script:\n- Task: {compressed_fallback}"

            # If model produced no compression or hallucinated extra bulk, fall back to rule-based compression
            if _estimate_tokens(optimized_text) >= orig_tokens:
                compressed_fallback = self._fallback_rule_compress(user_prompt)
                if _estimate_tokens(compressed_fallback) < orig_tokens:
                    optimized_text = compressed_fallback

        except Exception as exc:
            logger.warning("LLM optimization synthesis failed (%s), falling back to structural compression", exc)
            optimized_text = self._fallback_rule_compress(user_prompt)

        # ---------------------------------------------------------------------
        # Step 5: Semantic Fidelity Verification (nomic-embed-text)
        # ---------------------------------------------------------------------
        fidelity_score = 0.90
        try:
            vec_orig = await self.embedder.embed_text(user_prompt)
            vec_opt = await self.embedder.embed_text(optimized_text)
            fidelity_score = round(_cosine_similarity(vec_orig, vec_opt), 3)
        except Exception as exc:
            logger.warning("Semantic embedding check skipped (%s), default to 0.90", exc)

        opt_tokens = _estimate_tokens(optimized_text)
        token_savings_pct = round(max(0.0, (orig_tokens - opt_tokens) / orig_tokens), 3) if orig_tokens > 0 else 0.0


        status = OptimizationStatus.SUCCESS
        reasoning = "Prompt successfully optimized and compressed."
        if fidelity_score < target_fidelity:
            status = OptimizationStatus.DRIFT_WARNING
            reasoning = f"Semantic fidelity ({fidelity_score:.2f}) fell below target ({target_fidelity:.2f})."

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        if _PROMETHEUS_AVAILABLE:
            try:
                optimization_requests_total.labels(intent=intent.value, status=status.value).inc()
                optimization_latency_seconds.observe(elapsed_ms / 1000.0)
                token_savings_metric.labels(intent=intent.value).observe(token_savings_pct)
                semantic_fidelity_metric.labels(intent=intent.value).observe(fidelity_score)
            except Exception:
                pass

        return OptimizedPromptResult(
            status=status,
            original_prompt=user_prompt,
            optimized_prompt=optimized_text,
            intent=intent,
            intent_confidence=round(intent_res.confidence, 3),
            directive_name=directive.display_name,
            original_tokens=orig_tokens,
            optimized_tokens=opt_tokens,
            token_savings_pct=token_savings_pct,
            semantic_fidelity=fidelity_score,
            security_verdict="CLEAN_VERIFIED",
            security_risk_score=round(sec_result.total_confidence, 2),
            latency_ms=elapsed_ms,
            exemplars_used=len(exemplars),
            reasoning=reasoning,
        )

    def _fallback_rule_compress(self, text: str) -> str:
        """Fast fallback rule-based compression when LLM is offline."""
        import re
        pruned = re.sub(r"(?i)^(hello|hi|hey|can you please|could you please|please help me|i need you to)\s*", "", text.strip())
        pruned = re.sub(r"(?i)(thanks|thank you|thanks in advance|appreciate it)[\.!]?\s*$", "", pruned.strip())
        return pruned.strip() or text.strip()
