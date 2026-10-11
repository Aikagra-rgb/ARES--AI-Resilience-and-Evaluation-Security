"""
Pydantic v2 schemas for the ARES REST API.

All schemas are named and shaped to align 1-to-1 with the C# AresModels.cs
records in the Blazor frontend so JSON deserialization works without mapping.
"""

from __future__ import annotations

from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared enums (mirror C# enums)
# ---------------------------------------------------------------------------

class AiProviderEnum(str, Enum):
    openai    = "openai"
    groq      = "groq"
    gemini    = "gemini"
    nvidia_nim = "nvidia_nim"
    ollama    = "ollama"


class AttackCategoryEnum(str, Enum):
    direct_prompt_injection    = "direct_prompt_injection"
    indirect_prompt_injection  = "indirect_prompt_injection"
    system_prompt_extraction   = "system_prompt_extraction"
    data_exfiltration          = "data_exfiltration"
    policy_bypass              = "policy_bypass"
    role_manipulation          = "role_manipulation"
    tool_misuse                = "tool_misuse"
    encoding_or_obfuscation    = "encoding_or_obfuscation"


class TestRunStatusEnum(str, Enum):
    __test__ = False
    idle      = "idle"
    queued    = "queued"
    running   = "running"
    completed = "completed"
    blocked   = "blocked"
    failed    = "failed"
    cancelled = "cancelled"


class SeverityEnum(str, Enum):
    safe     = "safe"
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


class ProviderStatusEnum(str, Enum):
    operational    = "operational"
    degraded       = "degraded"
    unavailable    = "unavailable"
    not_configured = "not_configured"


class AttackSourceEnum(str, Enum):
    manual    = "manual"
    corpus    = "corpus"
    generated = "generated"


class IncidentStatusEnum(str, Enum):
    open          = "open"
    investigating = "investigating"
    resolved      = "resolved"
    suppressed    = "suppressed"


# ---------------------------------------------------------------------------
# Arena / Test schemas
# ---------------------------------------------------------------------------

class ArenaTestConfig(BaseModel):
    test_name:                  str   = "Security evaluation"
    target_application:         str   = "ARES Target"
    system_prompt:              str
    user_prompt:                str   = "Hello, can you help me?"
    provider:                   AiProviderEnum = AiProviderEnum.groq
    model:                      str   = "llama-3.1-8b-instant"
    attack_source:              AttackSourceEnum = AttackSourceEnum.corpus
    attack_categories:          List[AttackCategoryEnum] = [AttackCategoryEnum.role_manipulation]
    temperature:                float = Field(0.2, ge=0, le=2)
    max_response_tokens:        int   = Field(512, ge=64, le=4096)
    variation_count:            int   = Field(3, ge=1, le=10)
    include_hardened_comparison: bool = True


class CreateTestRequest(BaseModel):
    configuration: ArenaTestConfig


class CreateTestResponse(BaseModel):
    test_id:        str
    status:         TestRunStatusEnum
    correlation_id: str
    created_at:     datetime


class CancelTestResponse(BaseModel):
    test_id:        str
    status:         TestRunStatusEnum
    correlation_id: str


class DetectionResult(BaseModel):
    rule_id:     str
    name:        str
    severity:    SeverityEnum
    explanation: str
    triggered:   bool


class EvidenceItem(BaseModel):
    id:           str
    source:       str
    summary:      str
    similarity:   float
    category:     str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HardeningResult(BaseModel):
    summary:            str
    recommended_change: str
    hardened_prompt:    str
    improvement_points: int
    generated_at:       datetime


class ResponseComparison(BaseModel):
    baseline_response:  str
    hardened_response:  str
    difference_summary: str


class SecurityClassification(BaseModel):
    risk_score:             int
    severity:               SeverityEnum
    attack_succeeded:       bool
    runtime_classification: str
    detections:             List[DetectionResult] = []
    evidence:               List[EvidenceItem]    = []
    hardening:              Optional[HardeningResult]    = None
    comparison:             Optional[ResponseComparison] = None


class ExecutionLogEvent(BaseModel):
    timestamp: datetime
    stage:     str
    message:   str
    level:     str = "info"


class TestRunResponse(BaseModel):
    __test__ = False
    id:               str
    configuration:    ArenaTestConfig
    status:           TestRunStatusEnum
    created_at:       datetime
    completed_at:     Optional[datetime] = None
    duration_ms:      int = 0
    token_estimate:   int = 0
    analysis:         Optional[SecurityClassification] = None
    log:              List[ExecutionLogEvent] = []
    correlation_id:   str
    failure_reason:   Optional[str] = None


class RecentTestItem(BaseModel):
    id:         str
    timestamp:  datetime
    category:   AttackCategoryEnum
    provider:   AiProviderEnum
    model:      str
    risk_score: int
    status:     TestRunStatusEnum


# ---------------------------------------------------------------------------
# Dashboard schemas
# ---------------------------------------------------------------------------

class MetricValue(BaseModel):
    label:       str
    value:       str
    change:      str
    trend:       str
    status:      SeverityEnum
    description: str


class DashboardSummaryResponse(BaseModel):
    metrics:                 List[MetricValue]
    corpus_size:             int
    protected_applications:  int
    generated_at:            datetime
    is_partial:              bool = False


class TrendPoint(BaseModel):
    date:       str          # "YYYY-MM-DD"
    tested:     int
    blocked:    int
    successful: int
    incidents:  int


class CategoryMetricResponse(BaseModel):
    category:     str
    tests:        int
    successful:   int
    blocked:      int
    success_rate: int


class RuntimeIncident(BaseModel):
    id:                str
    application:       str
    category:          AttackCategoryEnum
    severity:          SeverityEnum
    detected_at:       datetime
    enforcement_action: str
    status:            IncidentStatusEnum
    correlation_id:    str


class ProviderHealthResponse(BaseModel):
    provider:   Optional[AiProviderEnum]
    name:       str
    status:     ProviderStatusEnum
    detail:     str
    checked_at: datetime


class HardeningComparison(BaseModel):
    baseline_success_rate:  int
    hardened_success_rate:  int
    improvement_points:     int
    tests_included:         int
    last_cycle:             str   # "YYYY-MM-DD"


class ModelConfigResponse(BaseModel):
    provider:       AiProviderEnum
    id:             str
    display_name:   str
    context_window: int
    is_available:   bool


# ---------------------------------------------------------------------------
# Corpus schemas
# ---------------------------------------------------------------------------

class CorpusSaveRequest(BaseModel):
    test_id:      Optional[str] = None
    attack_text:  Optional[str] = None
    category:     Optional[str] = None
    severity:     Optional[str] = None
    analyst_note: Optional[str] = None


class CorpusSaveResponse(BaseModel):
    attack_id:      str
    test_id:        str
    saved_at:       datetime
    correlation_id: str


# ---------------------------------------------------------------------------
# Harden schemas
# ---------------------------------------------------------------------------

class HardenRequest(BaseModel):
    system_prompt:    str
    application_name: str = "ARES App"
    domain:           str = "general"
    test_id:          Optional[str] = None
    optimize_tokens:  bool = True


class HardenResponse(BaseModel):
    hardened_prompt:     str
    baseline_score:      int
    hardened_score:      int
    improvement_points:  int
    strategy_applied:    str
    token_overhead:      int
    baseline_tokens:     Optional[int] = None
    hardened_tokens:     Optional[int] = None
    efficiency_score:    Optional[float] = None
    provider_used:       Optional[str] = "nvidia"
    model_used:          Optional[str] = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    ml_model_updated_at: Optional[datetime] = None
    generated_at:        datetime



# ---------------------------------------------------------------------------
# Search schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query:    str
    category: Optional[str] = None
    limit:    int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    id:               str
    score:            float
    attack_text:      str
    category:         str
    source:           str
    domain:           str
    severity:         str
    operator_applied: Optional[str] = None


class SearchResponse(BaseModel):
    query:      str
    total_hits: int
    hits:       List[SearchHit]


# ---------------------------------------------------------------------------
# ML Stats schemas
# ---------------------------------------------------------------------------

class CategoryAccuracy(BaseModel):
    category:  str
    precision: float
    recall:    float
    f1_score:  float
    support:   int


class StatsResponse(BaseModel):
    overall_accuracy:   float
    cv_accuracy:        float
    cv_std:             float
    corpus_size:        int
    category_counts:    Dict[str, int]
    category_breakdown: List[CategoryAccuracy]
    model_name:         str
    generated_at:       datetime


# ---------------------------------------------------------------------------
# Runtime Enforcement Gateway schemas
# ---------------------------------------------------------------------------

class EnforceActionEnum(str, Enum):
    allow = "ALLOW"
    sanitize = "SANITIZE"
    block = "BLOCK"


class EnforceRequest(BaseModel):
    prompt:           str
    system_prompt:    Optional[str] = None
    application_name: str = "ARES Protected Application"
    correlation_id:   Optional[str] = None


class EnforceResponse(BaseModel):
    action:            EnforceActionEnum
    risk_score:        int
    severity:          SeverityEnum
    matched_technique: Optional[str] = None
    vector_similarity: Optional[float] = None
    sanitized_prompt:  Optional[str] = None
    explanation:       str
    latency_ms:        float
    correlation_id:    str
    timestamp:         datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatProxyMessage(BaseModel):
    role:    str
    content: str


class ChatProxyRequest(BaseModel):
    model:            str = "groq"
    messages:         List[ChatProxyMessage]
    temperature:      float = 0.7
    application_name: str = "ARES Chat Proxy"


class ChatProxyResponse(BaseModel):
    action_taken:   EnforceActionEnum
    risk_score:     int
    content:        str
    model:          str
    latency_ms:     float
    correlation_id: str


class GatewayStatusResponse(BaseModel):
    status:                   str
    qdrant_points:            int
    block_threshold:          float
    sanitize_threshold:       float
    canary_token_configured:  bool
    version:                  str = "1.0.0"

