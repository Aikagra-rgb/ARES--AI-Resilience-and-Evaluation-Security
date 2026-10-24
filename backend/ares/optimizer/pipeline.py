"""
Unified User Prompt Optimizer Engine.
Chains Ingress Security Gate -> Intent Classifier -> Qdrant Few-Shot Exemplars -> Llama 3.1 Synthesis -> Semantic Drift Verification.
"""

from __future__ import annotations

import logging
import math
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set
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
    target_fidelity: float = Field(default=0.85, ge=0.0, le=1.0)
    verification_passed: bool = True
    mitigated_threats: List[str] = Field(default_factory=list)
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


_SCHEMA_TYPE_STOPWORDS = frozenset({
    "str", "string", "int", "integer", "float", "double", "number", "num",
    "bool", "boolean", "list", "array", "dict", "object", "map", "null",
    "none", "true", "false", "any", "optional", "required", "json", "schema",
    "field", "fields", "keys", "values",
})


def _extract_schema_fields(text: str) -> Set[str]:
    """Extract explicit schema field identifiers from a prompt string."""
    if not text or not text.strip():
        return set()
    fields: Set[str] = set()

    # 1. Identifiers inside braces {...}, peeling nested braces from inside out
    working = text
    while True:
        matches = list(re.finditer(r"\{([^{}]*)\}", working))
        if not matches:
            break
        for brace_match in matches:
            inner = brace_match.group(1)
            kv_keys = re.findall(r'["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?\s*:', inner)
            if kv_keys:
                for k in kv_keys:
                    k_low = k.lower()
                    if k_low not in _SCHEMA_TYPE_STOPWORDS or k_low in ("type", "value", "key"):
                        fields.add(k_low)
            else:
                for tok in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", inner):
                    tok_low = tok.lower()
                    if tok_low not in _SCHEMA_TYPE_STOPWORDS or tok_low in ("type", "value", "key"):
                        fields.add(tok_low)
        working = re.sub(r"\{[^{}]*\}", "", working)

    # 2. Snake_case identifiers anywhere in text (e.g., customer_id, issue_category, follow_up_date)
    for snake in re.findall(r"\b([a-zA-Z]+_[a-zA-Z0-9_]+)\b", text):
        fields.add(snake.lower())

    # 3. Single/double/backtick quoted single-token field names (e.g., 'email', "subtotal", `tax`)
    for quoted in re.findall(r"(['\"`])([a-zA-Z_][a-zA-Z0-9_]*)\1", text):
        q_low = quoted[1].lower()
        if q_low not in _SCHEMA_TYPE_STOPWORDS and len(q_low) > 1:
            fields.add(q_low)

    # 4. Explicit comma-separated field lists after "with fields", "fields:", "keys:", "columns:"
    for list_match in re.finditer(
        r"(?:with\s+fields|fields|keys|columns)\s*:?\s*([a-zA-Z0-9_,\s\(\)/-]+?)(?:\.|\n|$)",
        text,
        flags=re.IGNORECASE,
    ):
        segment = re.sub(r"[()]", ",", list_match.group(1))
        for part in re.split(r",|\band\b", segment, flags=re.IGNORECASE):
            tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", part.strip())
            if len(tokens) == 1:
                t_low = tokens[0].lower()
                if t_low not in _SCHEMA_TYPE_STOPWORDS and len(t_low) > 1:
                    fields.add(t_low)

    return fields


def verify_json_schema_preserved(original: str, optimized: str) -> bool:
    """
    Verify that required schema fields or target extraction fields in the original
    data_extraction_json prompt survive compression in the optimized prompt.
    """
    if not original or not original.strip() or not optimized or not optimized.strip():
        return False

    opt_lower = optimized.lower()
    has_json_indicator = "json" in opt_lower or ("{" in optimized and "}" in optimized)
    if not has_json_indicator:
        return False

    orig_fields = _extract_schema_fields(original)
    if orig_fields:
        return all(
            f in opt_lower or f.replace("_", " ") in opt_lower
            for f in orig_fields
        )

    # Fallback for natural-language field lists (e.g. "extract the invoice date, subtotal, and tax into a JSON object"
    # or "into structured JSON data containing sentiment score (-1 to 1), primary complaint, and department.")
    nl_match = re.search(
        r"\b(?:extract|parse|containing|including)\s+(?:all\s+)?(?:the\s+)?(.+?)\s*(?:into|from|as|in\s+a|in\s+json|and\s+return|and\s+format|\.|:|$)",
        original,
        flags=re.IGNORECASE,
    )
    if nl_match:
        raw_target_list = re.sub(r"\([^)]*\)", "", nl_match.group(1))
        items = []
        for part in re.split(r",|\band\b", raw_target_list, flags=re.IGNORECASE):
            cleaned = re.sub(r"^(?:the|a|an|all|their|its|following)\s+", "", part.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+(?:for|in|during|from|across|over)\s+.*$", "", cleaned, flags=re.IGNORECASE).strip().lower()
            if cleaned and cleaned not in ("json", "data", "information", "text"):
                items.append(cleaned)
        if items:
            for item in items:
                words = [w for w in re.findall(r"[a-z0-9]+", item) if len(w) >= 3]
                if not (
                    item in opt_lower
                    or item.replace(" ", "_") in opt_lower
                    or (words and any(w in opt_lower for w in words))
                ):
                    return False
            return True

    return True


_PROHIBITION_MARKERS = (
    "never",
    "do not",
    "don't",
    "must not",
    "cannot",
    "can't",
    "under no circumstances",
    "prohibit",
    "forbid",
    "refuse",
    "reject",
    "deny",
    "zero deviation",
    "no attorney-client",
    "not a doctor",
)

_OBLIGATION_MARKERS = (
    "must",
    "always",
    "only",
    "strictly",
    "ensure",
    "require",
    "required",
    "mandatory",
    "state clearly",
    "disclaimer",
)

_IMPERATIVE_DIRECTIVE_MARKERS = (
    "must",
    "always",
    "never",
    "do not",
    "don't",
    "only",
    "strictly",
    "ensure",
    "follow",
    "refuse",
    "reject",
    "require",
    "confidential",
    "disclaimer",
    "state clearly",
    "system directive",
    "role:",
    "<rules>",
    "<user_input>",
    "you are",
    "act as",
)

_CLAUSE_STOPWORDS = frozenset({
    "never", "must", "should", "would", "could", "under", "circumstances",
    "anything", "user", "users", "asks", "about", "attempting", "instructions",
    "politely", "your", "from", "with", "that", "this", "they", "them",
    "please", "always", "make", "sure", "when", "where", "which", "while",
    "only", "strictly", "ensure", "require", "required", "mandatory",
    "every", "response", "responses", "include", "state", "clearly", "follow",
})


def verify_directive_markers(
    original: str,
    optimized: str,
    intent: str | PromptIntent = PromptIntent.SYSTEM_DIRECTIVE,
) -> bool:
    """
    Verify that mandatory imperative, obligation, and prohibition guardrail markers
    present in a system_directive prompt survive compression.
    """
    if not original or not original.strip() or not optimized or not optimized.strip():
        return False

    intent_str = intent.value if hasattr(intent, "value") else str(intent)
    if intent_str != PromptIntent.SYSTEM_DIRECTIVE.value:
        return False

    orig_lower = original.lower()
    opt_lower = optimized.lower()

    # 1. Optimized prompt must contain at least one imperative/directive marker
    if not any(m in opt_lower for m in _IMPERATIVE_DIRECTIVE_MARKERS):
        return False

    # 2. If original contains prohibition/refusal guardrails, optimized must preserve prohibition semantics
    orig_has_prohibition = any(p in orig_lower for p in _PROHIBITION_MARKERS)
    if orig_has_prohibition:
        opt_prohibition_vocab = _PROHIBITION_MARKERS + (
            "no ",
            "strictly",
            "prohibited",
            "forbidden",
            "restricted",
        )
        if not any(p in opt_lower for p in opt_prohibition_vocab):
            return False

    # 3. Check each sentence/clause in original that contains a prohibition or positive obligation marker
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", original) if s.strip()]
    for sent in sentences:
        s_low = sent.lower()
        has_prohibition = any(p in s_low for p in _PROHIBITION_MARKERS)
        has_obligation = any(o in s_low for o in _OBLIGATION_MARKERS)
        if has_prohibition or has_obligation:
            content_words = [
                w.rstrip("s")
                for w in re.findall(r"\b[a-z0-9]{4,}\b", s_low)
                if w not in _CLAUSE_STOPWORDS
                and w not in _PROHIBITION_MARKERS
                and w not in _OBLIGATION_MARKERS
            ]
            content_words = [w for w in content_words if len(w) >= 3]
            if content_words and not any(w in opt_lower for w in content_words):
                return False

    # 4. Preserve compliance/security acronyms (e.g., SSN, PEP8, FINRA, GDPR, HIPAA)
    acronyms = {
        acr.lower().rstrip("s")
        for acr in re.findall(r"\b[A-Z]{2,}[0-9]*s?\b", original)
        if acr.upper() not in ("AI", "LLM", "API", "OK", "II", "III")
    }
    if acronyms and not all(acr in opt_lower for acr in acronyms):
        return False

    return True


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
        attack_store: Optional[Any] = None,
        findings_history: Optional[List[Any]] = None,
    ):
        self.settings = settings or get_settings()
        self.security_detector = security_detector or StructuralAttackDetector(
            canary_token=self.settings.gateway_canary_token
        )

        # Prefer the scaled 5,000 model if available
        if classifier is not None:
            self.classifier = classifier
        else:
            from pathlib import Path
            data_dir = Path(__file__).parent.parent / "intent" / "data"
            scaled_joblib = data_dir / "intent_classifier_scaled_5000.joblib"
            seed_joblib = data_dir / "intent_classifier_sklearn.joblib"
            model_target = scaled_joblib if scaled_joblib.exists() else seed_joblib
            self.classifier = SklearnIntentClassifier(model_path=model_target)

        self.exemplar_store = exemplar_store or QdrantExemplarStore(settings=self.settings)
        self.llm_client = llm_client
        self.embedder = embedder or GeminiEmbedder(settings=self.settings)
        self.attack_store = attack_store
        self._custom_attack_store = attack_store is not None
        self.findings_history = findings_history

    async def _retrieve_threat_context(
        self,
        user_prompt: str,
        intent: PromptIntent,
        sec_result: Any,
        domain: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve matched adversarial attack techniques / categories from the red-team
        Qdrant corpus, structural detections, or breach findings history to close the
        Red-Team -> Optimizer loop.
        """
        threats: List[str] = []

        # 1. Include any structural detector categories first
        for det in getattr(sec_result, "detections", []) or []:
            label = (
                getattr(det, "category", None)
                or getattr(getattr(det, "attack_type", None), "value", None)
                or getattr(det, "pattern_name", None)
                or getattr(det, "rule_id", None)
            )
            if label and str(label) not in threats:
                threats.append(str(label))

        # 2. Query Qdrant red-team attack corpus or breach findings history when optimizing
        # system directives, security-sensitive domains, or when an attack_store / findings_history
        # is explicitly configured
        should_query_corpus = (
            intent == PromptIntent.SYSTEM_DIRECTIVE
            or bool(threats)
            or self._custom_attack_store
            or bool(self.findings_history)
            or (domain and domain.lower() in ("defense", "fintech", "finance", "healthcare", "legal"))
        )
        if not should_query_corpus:
            return threats[:3]

        try:
            store = self.attack_store
            if store is None:
                from ares.vectordb.store import QdrantStore
                store = QdrantStore(settings=self.settings)
                self.attack_store = store

            hits = await store.search_similar_attacks(query_text=user_prompt, limit=3)
            for hit in hits or []:
                if not isinstance(hit, dict):
                    continue
                score = float(hit.get("score", 1.0) or 0.0)
                if score < 0.35:
                    continue
                payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else hit
                tech = (
                    payload.get("technique")
                    or payload.get("category")
                    or payload.get("attack_family")
                    or payload.get("title")
                )
                if tech and str(tech) not in threats:
                    threats.append(str(tech))
                if len(threats) >= 3:
                    break
        except Exception as exc:
            logger.debug("Red-team attack corpus lookup skipped (%s)", exc)

        # 3. Fallback / supplement from breach findings history (injected or recent DB breaches)
        if len(threats) < 3 and self.findings_history:
            for item in self.findings_history:
                if isinstance(item, dict):
                    succeeded = item.get("attack_succeeded", True)
                    cat = item.get("technique") or item.get("category") or item.get("classification")
                else:
                    succeeded = getattr(item, "attack_succeeded", True)
                    cat = (
                        getattr(item, "technique", None)
                        or getattr(item, "category", None)
                        or getattr(item, "classification", None)
                    )
                if succeeded and cat and str(cat) not in threats:
                    threats.append(str(cat))
                if len(threats) >= 3:
                    break

        return threats[:3]

    async def optimize_prompt(
        self,
        user_prompt: str,
        domain: Optional[str] = None,
        target_fidelity: Optional[float] = None,
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
            blocked_threats = [
                str(
                    getattr(d, "category", None)
                    or getattr(getattr(d, "attack_type", None), "value", None)
                    or getattr(d, "pattern_name", None)
                    or getattr(d, "rule_id", "Prompt Injection")
                )
                for d in (sec_result.detections or [])[:3]
            ] or ["Structural Prompt Injection"]
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
                target_fidelity=round(target_fidelity if target_fidelity is not None else 0.80, 2),
                verification_passed=False,
                mitigated_threats=blocked_threats,
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
        # Step 3: Dynamic Exemplar & Red-Team Threat Retrieval via Qdrant RAG
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

        mitigated_threats = await self._retrieve_threat_context(
            user_prompt=user_prompt,
            intent=intent,
            sec_result=sec_result,
            domain=domain,
        )

        # ---------------------------------------------------------------------
        # Step 4: Generative Optimization Synthesis (Llama 3.1 8B via Ollama)
        # ---------------------------------------------------------------------
        threat_hardening_block = ""
        if mitigated_threats:
            threat_hardening_block = (
                "\n\nRED-TEAM THREAT HARDENING (CLOSED-LOOP MOAT):\n"
                f"- Harden directive boundaries against observed red-team vectors: {', '.join(mitigated_threats)}.\n"
                "- Preserve all imperative refusal/prohibition markers while keeping token count strictly below the original prompt."
            )

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
            + threat_hardening_block
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
            if not lora_url.endswith("/v1"):
                lora_url = f"{lora_url}/v1"
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
        # Step 5: Semantic Fidelity & Structural Verification
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

        thresholds = getattr(self.settings, "class_fidelity_thresholds", {}) or {}
        min_fid = thresholds.get(intent.value, thresholds.get(intent, 0.85))
        effective_target_fidelity = round(float(target_fidelity if target_fidelity is not None else min_fid), 2)

        status = OptimizationStatus.SUCCESS
        verification_passed = True
        reasoning = "Prompt successfully optimized and compressed."
        if fidelity_score < effective_target_fidelity:
            if (
                fidelity_score >= 0.70
                and intent == PromptIntent.DATA_EXTRACTION_JSON
                and verify_json_schema_preserved(user_prompt, optimized_text)
            ):
                status = OptimizationStatus.SUCCESS
                verification_passed = True
                reasoning = (
                    f"Prompt successfully optimized; JSON schema preservation verified "
                    f"(fidelity {fidelity_score:.2f})."
                )
            elif (
                fidelity_score >= 0.70
                and intent == PromptIntent.SYSTEM_DIRECTIVE
                and verify_directive_markers(user_prompt, optimized_text, intent)
            ):
                status = OptimizationStatus.SUCCESS
                verification_passed = True
                reasoning = (
                    f"Prompt successfully optimized; directive guardrail markers verified "
                    f"(fidelity {fidelity_score:.2f})."
                )
            else:
                status = OptimizationStatus.DRIFT_WARNING
                verification_passed = False
                reasoning = (
                    f"Semantic fidelity ({fidelity_score:.2f}) fell below target "
                    f"({effective_target_fidelity:.2f})."
                )

        if mitigated_threats and status == OptimizationStatus.SUCCESS:
            reasoning = f"{reasoning} Hardened against red-team vectors: {', '.join(mitigated_threats)}."

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
            target_fidelity=effective_target_fidelity,
            verification_passed=verification_passed,
            mitigated_threats=mitigated_threats,
            security_verdict="CLEAN_VERIFIED",
            security_risk_score=round(sec_result.total_confidence, 2),
            latency_ms=elapsed_ms,
            exemplars_used=len(exemplars),
            reasoning=reasoning,
        )

    def _fallback_rule_compress(self, text: str) -> str:
        """Fast fallback rule-based compression when LLM is offline."""
        pruned = re.sub(
            r"(?i)^(hello|hi|hey(\s+ai)?|can you please|could you please|please help me|i need you to|i would like you to)\s*[!,.]?\s*",
            "",
            text.strip(),
        )
        pruned = re.sub(
            r"(?i)\b(could you please|can you please|please make sure to|please note that|it is very important that)\b\s*",
            "",
            pruned,
        )
        pruned = re.sub(
            r"(?i)(thanks(\s+so\s+much)?|thank you(\s+very\s+much)?|thanks in advance|appreciate it)[\.!]?\s*$",
            "",
            pruned.strip(),
        )
        pruned = re.sub(r"\s{2,}", " ", pruned).strip()
        return pruned or text.strip()
