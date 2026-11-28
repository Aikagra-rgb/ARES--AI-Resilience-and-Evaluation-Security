"""
Unified User Prompt Optimizer Engine.
Chains Ingress Security Gate -> Intent Classifier -> Qdrant Few-Shot Exemplars -> Llama 3.1 Synthesis -> Semantic Drift Verification.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from enum import Enum
from pathlib import Path
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
    VERIFICATION_UNAVAILABLE = "verification_unavailable"
    FALLBACK = "fallback"


class CompressionEngineKind(str, Enum):
    """Unambiguous optimizer synthesis path classification."""
    RULES_ADAPTER = "rules_adapter"
    CLIENT_LORA_ADAPTER = "client_lora_adapter"
    GLOBAL_LORA = "global_lora"
    BASE_MODEL_FALLBACK = "base_model_fallback"
    BLOCKED = "blocked"


_HIGH_ASSURANCE_INTENTS = frozenset({
    PromptIntent.DATA_EXTRACTION_JSON,
    PromptIntent.SYSTEM_DIRECTIVE,
    PromptIntent.CODING_TECHNICAL,
})


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
    verification_status: str = "verified"
    mitigated_threats: List[str] = Field(default_factory=list)
    security_verdict: str
    security_risk_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    exemplars_used: int = 0
    reasoning: Optional[str] = None
    org_id: str = "default"
    classifier_used: str = "global-sklearn"
    compression_engine: str = "rules_adapter:default"
    engine_kind: str = CompressionEngineKind.RULES_ADAPTER.value
    client_adapter_applied: bool = False
    client_lora_applied: bool = False
    protected_terms_preserved: bool = True
    missing_protected_terms: List[str] = Field(default_factory=list)


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


_TIKTOKEN_ENCODER: Any = None
_TIKTOKEN_CHECKED: bool = False


def _get_tiktoken_encoder() -> Any:
    """Lazily load the cl100k_base tiktoken encoder if installed and cached."""
    global _TIKTOKEN_ENCODER, _TIKTOKEN_CHECKED
    if not _TIKTOKEN_CHECKED:
        _TIKTOKEN_CHECKED = True
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODER = None
    return _TIKTOKEN_ENCODER


def _estimate_tokens(text: str) -> int:
    """
    Calculate subword token count using tiktoken (cl100k_base) when available,
    falling back to deterministic ~1.3 tokens per whitespace word heuristic.
    """
    if not text or not text.strip():
        return 0
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return max(1, len(encoder.encode(text, disallowed_special=())))
        except Exception:
            pass
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


def verify_protected_terms_preserved(
    original: str,
    optimized: str,
    protected_terms: Optional[List[str]],
) -> tuple[bool, List[str]]:
    """
    Verify that domain-critical protected terms from a client compression adapter
    that appeared in `original` survive in `optimized`.
    """
    if not original or not original.strip() or not protected_terms:
        return True, []
    if not optimized or not optimized.strip():
        present = [t.strip() for t in protected_terms if t and t.strip()]
        return (len(present) == 0), present

    def _contains_term(haystack: str, raw_term: str) -> bool:
        term = raw_term.strip()
        if not term:
            return True
        prefix = r"(?<!\w)" if (term[0].isalnum() or term[0] == "_") else ""
        suffix = r"(?!\w)" if (term[-1].isalnum() or term[-1] == "_") else ""
        pattern = prefix + re.escape(term) + suffix
        return bool(re.search(pattern, haystack, flags=re.IGNORECASE))

    missing: List[str] = []
    seen_lower: Set[str] = set()
    for raw_term in protected_terms:
        term = (raw_term or "").strip()
        if not term or term.lower() in seen_lower:
            continue
        seen_lower.add(term.lower())
        if _contains_term(original, raw_term) and not _contains_term(optimized, raw_term):
            missing.append(term)

    return len(missing) == 0, missing


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
        runtime_enforcer: Optional[Any] = None,
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
        self._runtime_enforcer = runtime_enforcer

    def get_runtime_enforcer(self) -> Any:
        """
        Lazily initialize and return the attached RuntimeEnforcer gateway engine.
        Attaches the pre-flight runtime sandbox directly to the optimizer in the background.
        """
        if self._runtime_enforcer is None:
            try:
                from ares.api.routes.gateway import _shared_enforcer
                if _shared_enforcer is not None:
                    self._runtime_enforcer = _shared_enforcer
            except Exception:
                pass

        if self._runtime_enforcer is None:
            try:
                from ares.gateway.enforcer import RuntimeEnforcer
                self._runtime_enforcer = RuntimeEnforcer(
                    settings=self.settings,
                    structural_detector=self.security_detector,
                )
            except Exception as exc:
                logger.debug("RuntimeEnforcer background initialization deferred: %s", exc)
        return self._runtime_enforcer

    async def _retrieve_threat_context(
        self,
        user_prompt: str,
        intent: PromptIntent,
        sec_result: Any,
        domain: Optional[str] = None,
        gateway_decision: Optional[Any] = None,
    ) -> List[str]:
        """
        Retrieve matched adversarial attack techniques / categories from the runtime
        gateway sandbox, red-team Qdrant corpus, structural detections, or breach findings
        history to close the Red-Team -> Optimizer loop.
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

    def _load_client_adapter_config(self, org_id: Optional[str]) -> Dict[str, Any]:
        """Load per-client compression adapter config and exemplars if trained on disk."""
        if not org_id or org_id == "default":
            return {}
        try:
            from ares.training.model_registry import MODELS_ROOT_DIR, validate_org_id

            safe_org = validate_org_id(org_id)
            root = MODELS_ROOT_DIR.resolve()
            adapter_dir = (root / safe_org / "compression_adapter").resolve()
            if root not in adapter_dir.parents:
                return {}
            cfg_path = adapter_dir / "adapter_config.json"
            meta_path = adapter_dir / "ares_adapter_metadata.json"
            if not cfg_path.exists() and not meta_path.exists():
                return {}
            cfg: Dict[str, Any] = {}
            if cfg_path.exists():
                cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
            if meta_path.exists():
                cfg.update(json.loads(meta_path.read_text(encoding="utf-8")))
            ex_path = adapter_dir / "exemplars.json"
            if ex_path.exists():
                cfg["_exemplars"] = json.loads(ex_path.read_text(encoding="utf-8"))
            return cfg
        except Exception as exc:
            logger.debug("Per-client adapter config skipped for '%s': %s", org_id, exc)
            return {}

    async def _post_process_llm_output(
        self,
        raw_out: Any,
        intent: PromptIntent,
        stage1_compressed: str,
        stage1_tokens: int,
        orig_tokens: int,
        engine_label: str,
        engine_kind: str,
        client_lora_applied: bool,
        fallback_engine_label: str,
    ) -> tuple[str, str, str, bool]:
        optimized_text = str(raw_out).strip()
        if optimized_text.startswith("```") and optimized_text.endswith("```"):
            lines = optimized_text.splitlines()
            if len(lines) >= 3 and intent != PromptIntent.DATA_EXTRACTION_JSON:
                optimized_text = "\n".join(lines[1:-1]).strip()

        for preamble in (
            "here is the optimized prompt:",
            "here is the compressed prompt:",
            "here is the rewritten prompt:",
            "optimized prompt:",
            "rewritten prompt:",
        ):
            if optimized_text.lower().startswith(preamble):
                optimized_text = optimized_text[len(preamble):].strip()

        if any(optimized_text.startswith(kw) for kw in ("def ", "async def ", "import ", "from ", "class ")):
            optimized_text = f"Write Python script:\n- Task: {stage1_compressed}"

        if _estimate_tokens(optimized_text) >= orig_tokens and stage1_tokens < orig_tokens:
            return stage1_compressed, fallback_engine_label, CompressionEngineKind.RULES_ADAPTER.value, False

        return optimized_text, engine_label, engine_kind, client_lora_applied

    async def _try_lora_rewrite(
        self,
        messages: List[ChatMessage],
        model_name: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> tuple[Optional[str], str, str, bool]:
        raw_lora_url = getattr(self.settings, "lora_server_url", "http://localhost:11435/v1") or "http://localhost:11435/v1"
        lora_url = raw_lora_url.strip().rstrip("/")
        if not lora_url.endswith("/v1"):
            lora_url = f"{lora_url}/v1"
        lora_model = model_name or "ares-optimizer-lora"
        payload: Dict[str, Any] = {
            "model": lora_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
            "max_tokens": 1000,
        }
        if org_id and org_id != "default":
            payload["org_id"] = org_id
        try:
            import httpx

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{lora_url}/chat/completions",
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_out = data["choices"][0]["message"]["content"]
                    used_adapter = str(data.get("adapter") or "default").strip()
                    if org_id and org_id != "default" and used_adapter == org_id:
                        logger.info("Optimized prompt via client LoRA adapter (%s)", org_id)
                        return (
                            raw_out,
                            f"client_lora_adapter:{org_id}",
                            CompressionEngineKind.CLIENT_LORA_ADAPTER.value,
                            True,
                        )
                    logger.info("Optimized prompt via global LoRA adapter (%s)", lora_model)
                    return (
                        raw_out,
                        f"global_lora:{lora_model}",
                        CompressionEngineKind.GLOBAL_LORA.value,
                        False,
                    )
        except Exception as lora_err:
            logger.debug("Native LoRA server offline (%s)", lora_err)
        return None, "", CompressionEngineKind.RULES_ADAPTER.value, False

    def _resolve_cloud_provider_and_model(
        self,
        model_name: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        cloud_order = (
            ("groq", getattr(self.settings, "groq_api_key", None), getattr(self.settings, "groq_default_model", None)),
            ("gemini", getattr(self.settings, "gemini_api_key", None), getattr(self.settings, "gemini_default_model", None)),
            ("openai", getattr(self.settings, "openai_api_key", None), getattr(self.settings, "openai_default_model", None)),
            ("nvidia", getattr(self.settings, "nvidia_api_key", None), getattr(self.settings, "nvidia_default_model", None)),
        )
        for prov_name, secret_val, default_model in cloud_order:
            if secret_val:
                return prov_name, model_name or default_model
        return None, None

    async def optimize_prompt(
        self,
        user_prompt: str,
        domain: Optional[str] = None,
        target_fidelity: Optional[float] = None,
        allow_harmful: bool = False,
        org_id: Optional[str] = None,
        prefer_rule_compression: bool = False,
        inference_mode: str = "auto",
        model_name: Optional[str] = None,
        pre_evaluated_decision: Optional[Any] = None,
    ) -> OptimizedPromptResult:
        """
        Execute full closed-loop prompt optimization pipeline:
        1. Pre-flight Runtime Gateway Sandbox & Structural Gate
        2. Per-Client / Global Intent Classification & Directive Resolution
        3. Dynamic Exemplar & Red-Team Threat Retrieval (+ Client Adapter Context)
        4. Two-Stage Synthesis: Stage-1 Lexical Pre-Pruning -> Stage-2 LoRA/LLM Refinement
        5. Per-Client Semantic Fidelity & Structural Verification
        6. Post-Optimization Gateway Enforcement (closed-loop canary & injection verification)
        """
        t0 = time.perf_counter()
        orig_tokens = _estimate_tokens(user_prompt)

        from ares.gateway.context import resolve_tenant_context
        from ares.training.model_registry import MODELS_ROOT_DIR, ModelRegistry, validate_org_id

        tenant_ctx = resolve_tenant_context(
            application_name=org_id,
            default_org_id=org_id or "default",
        )
        effective_org_id = org_id or tenant_ctx.organization_id
        safe_effective_org: Optional[str] = None
        if effective_org_id and effective_org_id != "default":
            try:
                safe_effective_org = validate_org_id(effective_org_id)
            except ValueError:
                safe_effective_org = None
        client_registry = ModelRegistry()
        client_cfg = client_registry.get_client_config(effective_org_id) or {}
        effective_canary = (
            tenant_ctx.canary_token
            or client_cfg.get("canary_token")
            or (self.settings.gateway_canary_token if self.settings.gateway_canary_token else None)
        )
        effective_domain = domain or client_cfg.get("domain")
        has_client_adapter = bool(
            safe_effective_org
            and (MODELS_ROOT_DIR / safe_effective_org / "compression_adapter" / "adapter_config.json").exists()
        )
        norm_mode = (inference_mode or "auto").strip().lower() or "auto"

        # ---------------------------------------------------------------------
        # Step 1: Pre-flight Runtime Gateway Sandbox & Security Gate
        # ---------------------------------------------------------------------
        gateway_decision = pre_evaluated_decision if not allow_harmful else None
        gateway_blocked = False
        prompt_to_optimize = user_prompt
        enforcer = self.get_runtime_enforcer()

        if gateway_decision is not None and not allow_harmful:
            action_val = getattr(getattr(gateway_decision, "action", None), "value", str(getattr(gateway_decision, "action", "")))
            risk_val = getattr(gateway_decision, "risk_score", 0) or 0
            if action_val == "BLOCK" or risk_val >= 85:
                gateway_blocked = True
            elif action_val == "SANITIZE" and getattr(gateway_decision, "sanitized_prompt", None):
                prompt_to_optimize = gateway_decision.sanitized_prompt
        elif enforcer is not None and not allow_harmful:
            try:
                from ares.api.schemas import EnforceRequest
                enforce_req = EnforceRequest(
                    prompt=user_prompt,
                    application_name=client_cfg.get("application_name") or f"ARES Prompt Optimizer ({effective_domain or 'default'})",
                )
                gateway_decision = await enforcer.enforce(
                    enforce_req,
                    context=tenant_ctx if effective_org_id != "default" else None,
                )
                action_val = getattr(getattr(gateway_decision, "action", None), "value", str(getattr(gateway_decision, "action", "")))
                risk_val = getattr(gateway_decision, "risk_score", 0) or 0
                if action_val == "BLOCK" or risk_val >= 85:
                    gateway_blocked = True
                elif action_val == "SANITIZE":
                    if getattr(gateway_decision, "sanitized_prompt", None):
                        prompt_to_optimize = gateway_decision.sanitized_prompt
            except Exception as gw_err:
                logger.warning("Runtime gateway sandbox enforcement error (%s); falling back to structural detector", gw_err)

        # Deterministic Layer-0 structural attack detection fallback
        sec_result = self.security_detector.detect(
            user_prompt,
            canary_token=effective_canary,
        )
        is_blocked = (gateway_blocked or sec_result.decision in (Decision.BLOCK, Decision.CHALLENGE)) and not allow_harmful

        if is_blocked:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            first_match = sec_result.detections[0] if sec_result.detections else None
            if gateway_decision and getattr(gateway_decision, "explanation", None):
                violation_msg = f"Blocked by Security Gate: {gateway_decision.explanation}"
                risk_score_norm = round(getattr(gateway_decision, "risk_score", 90) / 100.0, 2)
                tech = getattr(gateway_decision, "matched_technique", None) or "Adversarial Injection"
                blocked_threats = [str(tech)]
            elif first_match:
                violation_msg = f"Blocked by Security Gate: {first_match.description}"
                risk_score_norm = round(sec_result.total_confidence, 2)
                blocked_threats = [
                    str(
                        getattr(d, "category", None)
                        or getattr(getattr(d, "attack_type", None), "value", None)
                        or getattr(d, "pattern_name", None)
                        or getattr(d, "rule_id", "Prompt Injection")
                    )
                    for d in (sec_result.detections or [])[:3]
                ] or ["Structural Prompt Injection"]
            else:
                violation_msg = "Blocked by Security Gate: Structural prompt injection detected"
                risk_score_norm = round(sec_result.total_confidence, 2)
                blocked_threats = ["Structural Prompt Injection"]

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
                verification_status="blocked",
                mitigated_threats=blocked_threats,
                security_verdict="BLOCKED_ADVERSARIAL_PAYLOAD",
                security_risk_score=max(0.85, risk_score_norm),
                latency_ms=elapsed_ms,
                exemplars_used=0,
                reasoning=violation_msg,
                org_id=effective_org_id,
                classifier_used=(
                    f"per-client-{safe_effective_org}-v1"
                    if safe_effective_org
                    and (MODELS_ROOT_DIR / safe_effective_org / "intent_classifier").exists()
                    else "global-gateway"
                ),
                compression_engine="blocked",
                engine_kind=CompressionEngineKind.BLOCKED.value,
                client_adapter_applied=has_client_adapter,
                client_lora_applied=False,
            )

        # ---------------------------------------------------------------------
        # Step 2: Per-Client / Global Intent Classification & Directive Resolution
        # ---------------------------------------------------------------------
        client_classifier = None
        if effective_org_id and effective_org_id != "default":
            try:
                from ares.intent.per_client_classifier import get_per_client_classifier

                client_classifier = get_per_client_classifier(effective_org_id)
            except Exception:
                client_classifier = None

        if client_classifier is not None:
            intent_res = client_classifier.predict_intent(prompt_to_optimize)
            classifier_used = getattr(intent_res, "model_version", None) or f"per-client-{effective_org_id}-v1"
        else:
            intent_res = self.classifier.predict(prompt_to_optimize)
            classifier_used = getattr(intent_res, "model_version", None) or "global-sklearn"
        intent = intent_res.intent
        directive = intent_res.directive

        # ---------------------------------------------------------------------
        # Step 3: Dynamic Exemplar & Red-Team Threat Retrieval (+ Client Adapter)
        # ---------------------------------------------------------------------
        client_adapter_cfg = self._load_client_adapter_config(effective_org_id)
        client_adapter_applied = bool(client_adapter_cfg) or has_client_adapter
        exemplars = []
        few_shot_block = ""
        try:
            exemplars = await self.exemplar_store.search_exemplars(
                query_prompt=prompt_to_optimize,
                intent=intent,
                domain=effective_domain,
                limit=2,
            )
            few_shot_block = self.exemplar_store.format_few_shot_block(exemplars)
        except Exception as exc:
            logger.warning("Qdrant exemplar retrieval skipped (%s)", exc)

        client_exemplars = client_adapter_cfg.get("_exemplars") or []
        if not few_shot_block and client_exemplars:
            matching_ex = [
                ex for ex in client_exemplars if ex.get("intent") == intent.value
            ][:2] or client_exemplars[:2]
            few_shot_lines = ["CLIENT-SPECIFIC OPTIMIZATION EXEMPLARS:"]
            for ex in matching_ex:
                few_shot_lines.append(
                    f"Original: {ex.get('original_prompt', '')}\nOptimized: {ex.get('optimized_prompt', '')}"
                )
            few_shot_block = "\n\n".join(few_shot_lines)

        mitigated_threats = await self._retrieve_threat_context(
            user_prompt=user_prompt,
            intent=intent,
            sec_result=sec_result,
            domain=effective_domain,
            gateway_decision=gateway_decision,
        )

        # ---------------------------------------------------------------------
        # Step 4: Two-Stage Synthesis (Stage 1 Lexical Compaction -> Stage 2 LLM/LoRA)
        # ---------------------------------------------------------------------
        stage1_compressed = self._fallback_rule_compress(
            prompt_to_optimize,
            org_id=effective_org_id,
            adapter_cfg=client_adapter_cfg,
        )
        stage1_tokens = _estimate_tokens(stage1_compressed)
        stage1_savings_pct = (orig_tokens - stage1_tokens) / orig_tokens if orig_tokens > 0 else 0.0

        threat_hardening_block = ""
        if mitigated_threats:
            threat_hardening_block = (
                "\n\nRED-TEAM THREAT HARDENING (CLOSED-LOOP MOAT):\n"
                f"- Harden directive boundaries against observed red-team vectors: {', '.join(mitigated_threats)}.\n"
                "- Preserve all imperative refusal/prohibition markers while keeping token count strictly below the original prompt."
            )

        protected_vocab_block = ""
        raw_protected_terms = client_adapter_cfg.get("protected_terms") or []
        active_protected_terms = [
            t.strip()
            for t in raw_protected_terms
            if t and t.strip() and t.strip().lower() in prompt_to_optimize.lower()
        ]
        if active_protected_terms:
            protected_vocab_block = (
                "\n\nPROTECTED CLIENT VOCABULARY (MUST PRESERVE VERBATIM):\n"
                f"- Preserve these exact domain terms if present: {', '.join(active_protected_terms)}."
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
            + protected_vocab_block
            + f"\n\nOUTPUT FORMAT:\n"
            f"Output ONLY the rewritten, compressed prompt text. Do NOT prepend explanations, introductions, or pleasantries."
        )

        user_content_parts = []
        if few_shot_block:
            user_content_parts.append(few_shot_block)
        user_content_parts.append(f"ORIGINAL UNOPTIMIZED PROMPT:\n{prompt_to_optimize}\n\nREWRITTEN OPTIMIZED PROMPT:")
        user_message_text = "\n\n".join(user_content_parts)

        messages = [
            ChatMessage(role="system", content=system_instruction),
            ChatMessage(role="user", content=user_message_text),
        ]

        if self.llm_client is None:
            self.llm_client = LLMClient()

        rules_engine_label = (
            f"rules_adapter:{effective_org_id}" if client_adapter_applied else "rules_adapter:default"
        )
        optimized_text = ""
        compression_engine = rules_engine_label
        engine_kind = CompressionEngineKind.RULES_ADAPTER.value
        client_lora_applied = False

        if norm_mode in ("client_adapter", "rules_adapter"):
            optimized_text = stage1_compressed
            compression_engine = rules_engine_label
            engine_kind = CompressionEngineKind.RULES_ADAPTER.value
            client_lora_applied = False
        elif norm_mode == "auto" and prefer_rule_compression and stage1_savings_pct >= 0.25:
            optimized_text = stage1_compressed
            compression_engine = rules_engine_label
            engine_kind = CompressionEngineKind.RULES_ADAPTER.value
            client_lora_applied = False
        else:
            try:
                raw_out = None
                chosen_engine = rules_engine_label
                chosen_kind = CompressionEngineKind.RULES_ADAPTER.value
                chosen_client_lora = False

                if norm_mode == "lora":
                    raw_out, chosen_engine, chosen_kind, chosen_client_lora = await self._try_lora_rewrite(
                        messages, model_name=model_name, org_id=effective_org_id
                    )
                elif norm_mode == "cloud":
                    cloud_prov, cloud_model = self._resolve_cloud_provider_and_model(model_name=model_name)
                    if cloud_prov:
                        resp = await self.llm_client.complete(
                            messages=messages,
                            provider=cloud_prov,
                            model=cloud_model,
                            temperature=0.2,
                            max_tokens=1000,
                        )
                        raw_out = getattr(resp, "text", None) or getattr(resp, "content", "")
                        chosen_engine = f"base_model_fallback:{cloud_prov}:{cloud_model}"
                        chosen_kind = CompressionEngineKind.BASE_MODEL_FALLBACK.value
                        chosen_client_lora = False
                elif norm_mode == "ollama":
                    primary_model = model_name or getattr(self.settings, "ollama_optimizer_model", "ares-optimizer:8b") or "ares-optimizer:8b"
                    fallback_model = self.settings.hardener_model or "llama3.1:8b"
                    try:
                        resp = await self.llm_client.complete(
                            messages=messages,
                            provider=LLMProvider.OLLAMA.value,
                            model=primary_model,
                            temperature=0.2,
                            max_tokens=1000,
                        )
                        used_model = primary_model
                    except Exception as model_err:
                        logger.warning("Primary Ollama optimizer '%s' unavailable (%s); falling back to '%s'", primary_model, model_err, fallback_model)
                        resp = await self.llm_client.complete(
                            messages=messages,
                            provider=LLMProvider.OLLAMA.value,
                            model=fallback_model,
                            temperature=0.2,
                            max_tokens=1000,
                        )
                        used_model = fallback_model
                    raw_out = getattr(resp, "text", None) or getattr(resp, "content", "")
                    if used_model == primary_model and "ares-optimizer" in used_model.lower():
                        chosen_engine = f"global_lora:{used_model}"
                        chosen_kind = CompressionEngineKind.GLOBAL_LORA.value
                    else:
                        chosen_engine = f"base_model_fallback:ollama:{used_model}"
                        chosen_kind = CompressionEngineKind.BASE_MODEL_FALLBACK.value
                    chosen_client_lora = False
                else:
                    # "auto" (default): Native LoRA -> configured provider/Ollama
                    use_lora = getattr(self.settings, "use_lora_server", True)
                    if use_lora:
                        raw_out, chosen_engine, chosen_kind, chosen_client_lora = await self._try_lora_rewrite(
                            messages, model_name=model_name, org_id=effective_org_id
                        )

                    if raw_out is None:
                        provider = self.settings.hardener_provider or LLMProvider.OLLAMA
                        prov_str = provider.value if isinstance(provider, LLMProvider) else str(provider)
                        primary_model = model_name or getattr(self.settings, "ollama_optimizer_model", "ares-optimizer:8b") or "ares-optimizer:8b"
                        fallback_model = self.settings.hardener_model or "llama3.1:8b"
                        try:
                            resp = await self.llm_client.complete(
                                messages=messages,
                                provider=prov_str,
                                model=primary_model,
                                temperature=0.2,
                                max_tokens=1000,
                            )
                            used_model = primary_model
                        except Exception as model_err:
                            logger.warning("Primary optimizer '%s' unavailable (%s); falling back to base '%s'", primary_model, model_err, fallback_model)
                            resp = await self.llm_client.complete(
                                messages=messages,
                                provider=prov_str,
                                model=fallback_model,
                                temperature=0.2,
                                max_tokens=1000,
                            )
                            used_model = fallback_model
                        raw_out = getattr(resp, "text", None) or getattr(resp, "content", "")
                        if used_model == primary_model and "ares-optimizer" in used_model.lower():
                            chosen_engine = f"global_lora:{used_model}"
                            chosen_kind = CompressionEngineKind.GLOBAL_LORA.value
                        else:
                            chosen_engine = f"base_model_fallback:{prov_str}:{used_model}"
                            chosen_kind = CompressionEngineKind.BASE_MODEL_FALLBACK.value
                        chosen_client_lora = False

                if raw_out is not None and str(raw_out).strip():
                    optimized_text, compression_engine, engine_kind, client_lora_applied = (
                        await self._post_process_llm_output(
                            raw_out=raw_out,
                            intent=intent,
                            stage1_compressed=stage1_compressed,
                            stage1_tokens=stage1_tokens,
                            orig_tokens=orig_tokens,
                            engine_label=chosen_engine,
                            engine_kind=chosen_kind,
                            client_lora_applied=chosen_client_lora,
                            fallback_engine_label=rules_engine_label,
                        )
                    )
                else:
                    optimized_text = stage1_compressed
                    compression_engine = rules_engine_label
                    engine_kind = CompressionEngineKind.RULES_ADAPTER.value
                    client_lora_applied = False
            except Exception as exc:
                logger.warning("LLM optimization synthesis failed (%s), falling back to Stage-1 rules compression", exc)
                optimized_text = stage1_compressed
                compression_engine = rules_engine_label
                engine_kind = CompressionEngineKind.RULES_ADAPTER.value
                client_lora_applied = False

        # ---------------------------------------------------------------------
        # Step 5: Fail-Closed Semantic Fidelity & Structural Verification
        # ---------------------------------------------------------------------
        fidelity_available = False
        fidelity_score = 0.0
        try:
            vec_orig = await self.embedder.embed_text(user_prompt)
            vec_opt = await self.embedder.embed_text(optimized_text)
            if vec_orig and vec_opt and len(vec_orig) == len(vec_opt):
                fidelity_score = round(_cosine_similarity(vec_orig, vec_opt), 3)
                fidelity_available = True
        except Exception as exc:
            logger.warning("Semantic fidelity verification unavailable (%s); failing closed", exc)

        opt_tokens = _estimate_tokens(optimized_text)
        token_savings_pct = round(max(0.0, (orig_tokens - opt_tokens) / orig_tokens), 3) if orig_tokens > 0 else 0.0

        thresholds = getattr(self.settings, "class_fidelity_thresholds", {}) or {}
        global_min_fid = float(thresholds.get(intent.value, thresholds.get(intent, 0.85)))
        min_fid = (
            client_registry.get_fidelity_threshold(effective_org_id, intent.value, fallback=global_min_fid)
            if effective_org_id and effective_org_id != "default"
            else global_min_fid
        )
        effective_target_fidelity = round(float(target_fidelity if target_fidelity is not None else min_fid), 2)

        # ---------------------------------------------------------------------
        # Step 6: Post-Optimization Gateway Security Check (Closed Loop)
        # ---------------------------------------------------------------------
        if not allow_harmful and optimized_text:
            post_struct = self.security_detector.detect(optimized_text, canary_token=effective_canary)
            canary_hit = bool(effective_canary and effective_canary in optimized_text)
            if post_struct.decision in (Decision.BLOCK, Decision.CHALLENGE) or canary_hit:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                return OptimizedPromptResult(
                    status=OptimizationStatus.BLOCKED_SECURITY,
                    original_prompt=user_prompt,
                    optimized_prompt=None,
                    intent=intent,
                    intent_confidence=round(intent_res.confidence, 3),
                    directive_name="Post-Optimization Gateway Quarantine",
                    original_tokens=orig_tokens,
                    optimized_tokens=0,
                    token_savings_pct=0.0,
                    semantic_fidelity=0.0,
                    target_fidelity=effective_target_fidelity,
                    verification_passed=False,
                    verification_status="blocked",
                    mitigated_threats=mitigated_threats or ["post_optimization_policy_violation"],
                    security_verdict="BLOCKED_POST_OPTIMIZATION_GATEWAY",
                    security_risk_score=0.95,
                    latency_ms=elapsed_ms,
                    exemplars_used=len(exemplars),
                    reasoning="Post-optimization gateway check blocked unsafe or canary-leaking rewritten output.",
                    org_id=effective_org_id,
                    classifier_used=classifier_used,
                    compression_engine="blocked",
                    engine_kind=CompressionEngineKind.BLOCKED.value,
                    client_adapter_applied=client_adapter_applied,
                    client_lora_applied=False,
                )

        # Verify domain-critical protected_terms from client rules configuration
        raw_adapter_terms = client_adapter_cfg.get("protected_terms") or []
        terms_ok, missing_terms = verify_protected_terms_preserved(
            prompt_to_optimize,
            optimized_text,
            raw_adapter_terms,
        )
        if not terms_ok and optimized_text != stage1_compressed:
            stage1_ok, _ = verify_protected_terms_preserved(
                prompt_to_optimize,
                stage1_compressed,
                raw_adapter_terms,
            )
            if stage1_ok:
                optimized_text = stage1_compressed
                opt_tokens = stage1_tokens
                token_savings_pct = (
                    round(max(0.0, (orig_tokens - opt_tokens) / orig_tokens), 3)
                    if orig_tokens > 0
                    else 0.0
                )
                compression_engine = rules_engine_label
                engine_kind = CompressionEngineKind.RULES_ADAPTER.value
                client_lora_applied = False
                terms_ok = True
                missing_terms = []

        final_optimized_text: Optional[str] = optimized_text
        if not fidelity_available:
            status = OptimizationStatus.VERIFICATION_UNAVAILABLE
            verification_passed = False
            verification_status = "verification_unavailable"
            llm_rewrite_used = engine_kind in (
                CompressionEngineKind.CLIENT_LORA_ADAPTER.value,
                CompressionEngineKind.GLOBAL_LORA.value,
                CompressionEngineKind.BASE_MODEL_FALLBACK.value,
            )
            if llm_rewrite_used:
                if intent in _HIGH_ASSURANCE_INTENTS:
                    final_optimized_text = None
                    opt_tokens = 0
                    token_savings_pct = 0.0
                    reasoning = (
                        "Semantic fidelity verification unavailable; rejected unverified LLM rewrite "
                        "for high-assurance prompt intent."
                    )
                else:
                    stage1_terms_ok, stage1_missing = verify_protected_terms_preserved(
                        prompt_to_optimize,
                        stage1_compressed,
                        raw_adapter_terms,
                    )
                    terms_ok = stage1_terms_ok
                    missing_terms = stage1_missing
                    compression_engine = rules_engine_label
                    engine_kind = CompressionEngineKind.RULES_ADAPTER.value
                    client_lora_applied = False
                    if stage1_terms_ok:
                        final_optimized_text = stage1_compressed
                        opt_tokens = stage1_tokens
                        token_savings_pct = (
                            round(max(0.0, (orig_tokens - stage1_tokens) / orig_tokens), 3)
                            if orig_tokens > 0
                            else 0.0
                        )
                        reasoning = (
                            "Semantic fidelity verification unavailable; reverted LLM rewrite to "
                            "deterministic Stage-1 rules compaction."
                        )
                    else:
                        final_optimized_text = None
                        opt_tokens = 0
                        token_savings_pct = 0.0
                        reasoning = (
                            "Semantic fidelity verification unavailable and Stage-1 rules dropped "
                            "protected terms; prompt rewrite rejected."
                        )
            else:
                structural_ok = terms_ok
                if structural_ok and intent == PromptIntent.DATA_EXTRACTION_JSON:
                    structural_ok = verify_json_schema_preserved(user_prompt, stage1_compressed)
                elif structural_ok and intent == PromptIntent.SYSTEM_DIRECTIVE:
                    structural_ok = verify_directive_markers(user_prompt, stage1_compressed, intent)
                if structural_ok:
                    final_optimized_text = stage1_compressed
                    opt_tokens = stage1_tokens
                    token_savings_pct = (
                        round(max(0.0, (orig_tokens - stage1_tokens) / orig_tokens), 3)
                        if orig_tokens > 0
                        else 0.0
                    )
                    reasoning = (
                        "Semantic fidelity verification unavailable; retained deterministic Stage-1 "
                        "rules compaction after structural checks."
                    )
                else:
                    final_optimized_text = None
                    opt_tokens = 0
                    token_savings_pct = 0.0
                    reasoning = (
                        "Semantic fidelity verification unavailable and structural invariants failed; "
                        "prompt rewrite rejected."
                    )
        else:
            status = OptimizationStatus.SUCCESS
            verification_passed = True
            verification_status = "verified"
            reasoning = "Prompt successfully optimized and compressed."
            if fidelity_score < effective_target_fidelity:
                if (
                    fidelity_score >= 0.70
                    and intent == PromptIntent.DATA_EXTRACTION_JSON
                    and verify_json_schema_preserved(user_prompt, optimized_text)
                ):
                    status = OptimizationStatus.SUCCESS
                    verification_passed = True
                    verification_status = "verified"
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
                    verification_status = "verified"
                    reasoning = (
                        f"Prompt successfully optimized; directive guardrail markers verified "
                        f"(fidelity {fidelity_score:.2f})."
                    )
                else:
                    status = OptimizationStatus.DRIFT_WARNING
                    verification_passed = False
                    verification_status = "drift_detected"
                    reasoning = (
                        f"Semantic fidelity ({fidelity_score:.2f}) fell below target "
                        f"({effective_target_fidelity:.2f})."
                    )

            if not terms_ok:
                status = OptimizationStatus.DRIFT_WARNING
                verification_passed = False
                verification_status = "drift_detected"
                reasoning = f"{reasoning} Missing protected client terms: {', '.join(missing_terms)}."

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

        sec_verdict = "CLEAN_VERIFIED"
        action_str = getattr(getattr(gateway_decision, "action", None), "value", str(getattr(gateway_decision, "action", ""))) if gateway_decision else ""
        if action_str == "SANITIZE":
            sec_verdict = "SANITIZED_BOUNDARY_ISOLATED"

        if action_str == "ALLOW" or not gateway_decision:
            gw_risk = 0.0
        else:
            gw_risk = (getattr(gateway_decision, "risk_score", 0) or 0) / 100.0

        return OptimizedPromptResult(
            status=status,
            original_prompt=user_prompt,
            optimized_prompt=final_optimized_text,
            intent=intent,
            intent_confidence=round(intent_res.confidence, 3),
            directive_name=directive.display_name,
            original_tokens=orig_tokens,
            optimized_tokens=opt_tokens,
            token_savings_pct=token_savings_pct,
            semantic_fidelity=fidelity_score,
            target_fidelity=effective_target_fidelity,
            verification_passed=verification_passed,
            verification_status=verification_status,
            mitigated_threats=mitigated_threats,
            security_verdict=sec_verdict,
            security_risk_score=round(max(sec_result.total_confidence, gw_risk), 2),
            latency_ms=elapsed_ms,
            exemplars_used=len(exemplars),
            reasoning=reasoning,
            org_id=effective_org_id,
            classifier_used=classifier_used,
            compression_engine=compression_engine,
            engine_kind=engine_kind,
            client_adapter_applied=client_adapter_applied,
            client_lora_applied=client_lora_applied,
            protected_terms_preserved=terms_ok,
            missing_protected_terms=missing_terms,
        )

    async def optimize_prompt_stream(
        self,
        user_prompt: str,
        domain: Optional[str] = None,
        target_fidelity: Optional[float] = None,
        allow_harmful: bool = False,
        org_id: Optional[str] = None,
        prefer_rule_compression: bool = False,
        inference_mode: str = "auto",
        model_name: Optional[str] = None,
    ):
        """
        Async generator yielding real-time SSE stage telemetry dicts followed by the final result.
        """
        yield {
            "event": "stage",
            "stage": "gateway_sandbox",
            "step": 1,
            "total_steps": 5,
            "status": "running",
            "message": "Evaluating inbound prompt in Runtime Gateway Sandbox...",
        }
        result = await self.optimize_prompt(
            user_prompt=user_prompt,
            domain=domain,
            target_fidelity=target_fidelity,
            allow_harmful=allow_harmful,
            org_id=org_id,
            prefer_rule_compression=prefer_rule_compression,
            inference_mode=inference_mode,
            model_name=model_name,
        )
        if result.status == OptimizationStatus.BLOCKED_SECURITY:
            yield {
                "event": "stage",
                "stage": "gateway_sandbox",
                "step": 1,
                "total_steps": 5,
                "status": "blocked",
                "message": result.reasoning or "Blocked by Runtime Gateway Sandbox.",
            }
            yield {
                "event": "complete",
                "stage": "complete",
                "result": result.model_dump(mode="json"),
            }
            return

        yield {
            "event": "stage",
            "stage": "gateway_sandbox",
            "step": 1,
            "total_steps": 5,
            "status": "completed",
            "message": f"Runtime Gateway Verdict: {result.security_verdict}",
        }
        yield {
            "event": "stage",
            "stage": "intent_classification",
            "step": 2,
            "total_steps": 5,
            "status": "completed",
            "message": f"Classified intent as '{result.intent.value}' ({int(result.intent_confidence * 100)}% confidence) -> {result.directive_name}",
        }
        yield {
            "event": "stage",
            "stage": "qdrant_rag",
            "step": 3,
            "total_steps": 5,
            "status": "completed",
            "message": f"Retrieved {result.exemplars_used} Qdrant exemplars and {len(result.mitigated_threats)} threat vectors",
        }
        yield {
            "event": "stage",
            "stage": "llm_synthesis",
            "step": 4,
            "total_steps": 5,
            "status": "completed",
            "message": f"Compressed {result.original_tokens} -> {result.optimized_tokens} tokens ({int(result.token_savings_pct * 100)}% reduction)",
        }
        verify_msg = (
            "Semantic verification unavailable — failed closed"
            if result.status == OptimizationStatus.VERIFICATION_UNAVAILABLE
            else f"Semantic fidelity {result.semantic_fidelity:.3f} vs {result.target_fidelity:.2f} target ({'PASSED' if result.verification_passed else 'FALLBACK'})"
        )
        yield {
            "event": "stage",
            "stage": "fidelity_verification",
            "step": 5,
            "total_steps": 5,
            "status": "completed" if result.verification_passed else "warning",
            "message": verify_msg,
        }
        yield {
            "event": "complete",
            "stage": "complete",
            "result": result.model_dump(mode="json"),
        }

    def _fallback_rule_compress(
        self,
        text: str,
        org_id: Optional[str] = None,
        adapter_cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fast Stage-1 lexical compression (<1ms).
        Strips conversational greetings, polite hedges, wordy transitional phrases,
        and trailing gratitude, while applying per-client domain replacements.
        """
        pruned = text.strip()
        # 1. Strip conversational opening frames
        opening_patterns = [
            r"(?i)^(hello|hi(\s+there)?|hey(\s+there)?(\s+ai)?|yo|good\s+(morning|afternoon|evening))[\s!,.]*",
            r"(?i)^(i\s+was\s+wondering\s+if\s+you\s+could(\s+possibly)?|could\s+you\s+please(\s+help\s+me(\s+out)?(\s+with)?)?|can\s+you\s+please(\s+help\s+me(\s+out)?(\s+with)?)?|would\s+you\s+mind|please\s+help\s+me(\s+with)?|i\s+need\s+you\s+to|i\s+would\s+(really\s+)?appreciate\s+it\s+if\s+you\s+could|i\s+would\s+like\s+you\s+to)\s+",
            r"(?i)^(i\s+have\s+been\s+working\s+on\s+this\s+all\s+\w+\s+and\s+really\s+need\s+some\s+help\.?\s*(here\s+is\s+what\s+i\s+am\s+trying\s+to\s+do:)?\s*)",
            r"(?i)^(from\s+an\s+enterprise\s+architecture\s+and\s+compliance\s+perspective,?\s*please\s+address:\s*)",
            r"(?i)^(i\s+am\s+trying\s+to\s+learn\s+more\s+about\s+how\s+to\s+handle\s+this\s+properly\.?\s*can\s+you\s+explain\s+and\s+demonstrate:\s*)",
            r"(?i)^(working\s+on\s+a\s+\w+\s+project,?\s*|in\s+our\s+\w+\s+application,?\s*|for\s+a\s+production\s+\w+\s+system,?\s*|regarding\s+our\s+\w+\s+service,?\s*)",
        ]
        for _ in range(3):
            before = pruned
            for pat in opening_patterns:
                pruned = re.sub(pat, "", pruned).strip()
            if pruned == before:
                break

        # 2. Inline verbose padding & clause compaction
        inline_replacements = [
            (r"(?i)\b(could\s+you\s+please|can\s+you\s+please|please\s+make\s+sure(\s+that|\s+to)|please\s+note\s+that|it\s+is\s+very\s+important\s+that|if\s+it\s+is\s+not\s+too\s+much\s+trouble)\b[\s,]*", ""),
            (r"(?i)\b(to\s+the\s+best\s+of\s+your\s+ability|when\s+you\s+get\s+a\s+chance|as\s+soon\s+as\s+possible)\b[\s,]*", ""),
            (r"(?i)\bdue\s+to\s+the\s+fact\s+that\b", "because"),
            (r"(?i)\bin\s+order\s+to\b", "to"),
            (r"(?i)\bfor\s+the\s+purpose\s+of\b", "for"),
            (r"(?i)\bat\s+this\s+point\s+in\s+time\b", "now"),
            (r"(?i)\bin\s+the\s+event\s+that\b", "if"),
            (r"(?i)\ba\s+large\s+number\s+of\b", "many"),
        ]
        for pat, repl in inline_replacements:
            pruned = re.sub(pat, repl, pruned)

        # 3. Client-specific lexical rules from compression_adapter/adapter_config.json
        cfg = adapter_cfg if adapter_cfg is not None else self._load_client_adapter_config(org_id)
        protected_terms = [
            t.strip().lower() for t in (cfg.get("protected_terms") or []) if t and t.strip()
        ]
        text_lower = text.lower()
        active_protected = [pt for pt in protected_terms if pt in text_lower]
        for src_phrase, dst_phrase in (cfg.get("lexical_replacements") or {}).items():
            src_low = src_phrase.lower()
            dst_low = dst_phrase.lower()
            if any(pt in src_low and pt not in dst_low for pt in active_protected):
                continue
            pruned = re.sub(r"(?i)" + re.escape(src_phrase), dst_phrase, pruned)

        # 4. Strip trailing conversational sign-offs
        trailing_patterns = [
            r"(?i)(please\s+make\s+sure\s+this\s+is\s+accurate,?\s*complete,?\s*and\s+formatted\s+well\.?\s*)$",
            r"(?i)(please\s+provide\s+the\s+answer\s+right\s+away\s+without\s+unnecessary\s+filler\.?\s*)$",
            r"(?i)(ensure\s+the\s+output\s+aligns\s+with\s+production\s+best\s+practices\.?\s*)$",
            r"(?i)(thank\s+you(\s+so|\s+very)?\s+much(\s+in\s+advance)?(\s+for\s+your\s+(help|assistance|time))?[!.]*\s*)$",
            r"(?i)(thanks(\s+so\s+much|\s+a\s+lot)?(\s+in\s+advance)?[!.]*\s*|cheers[!.]*\s*|appreciate\s+it[!.]*\s*)$",
        ]
        for _ in range(3):
            before = pruned
            for pat in trailing_patterns:
                pruned = re.sub(pat, "", pruned.strip()).strip()
            if pruned == before:
                break

        pruned = re.sub(r"\s+([,.;:!?])", r"\1", pruned)
        pruned = re.sub(r"\s{2,}", " ", pruned).strip(" ,")
        return pruned or text.strip()
