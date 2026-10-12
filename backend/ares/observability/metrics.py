"""
ARES Observability Module.

Provides Prometheus metrics, structured logging, and distributed tracing
integration for the ARES system.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ares.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class MetricsRegistry:
    """Container for all ARES metrics."""
    # Store registry for metrics generation
    _registry: Optional[Any] = None
    
    # Counters
    gateway_requests_total: Optional[Any] = None
    gateway_blocks_total: Optional[Any] = None
    gateway_sanitizes_total: Optional[Any] = None
    gateway_allows_total: Optional[Any] = None
    
    # Structural detector
    structural_detections_total: Optional[Any] = None
    structural_blocks_total: Optional[Any] = None
    
    # Vector similarity
    vector_searches_total: Optional[Any] = None
    vector_blocks_total: Optional[Any] = None
    
    # ML Classifier
    ml_classifications_total: Optional[Any] = None
    
    # Consensus
    consensus_decisions_total: Optional[Any] = None
    consensus_conflicts_total: Optional[Any] = None
    
    # Audit logging
    audit_events_total: Optional[Any] = None
    audit_failures_total: Optional[Any] = None
    
    # LLM Provider calls
    llm_calls_total: Optional[Any] = None
    llm_call_failures_total: Optional[Any] = None
    llm_fallback_total: Optional[Any] = None
    
    # Threat intel
    threat_intel_fetched_total: Optional[Any] = None
    threat_intel_indexed_total: Optional[Any] = None
    threat_intel_rejected_total: Optional[Any] = None
    
    # Moderation queue
    moderation_submitted_total: Optional[Any] = None
    moderation_approved_total: Optional[Any] = None
    moderation_rejected_total: Optional[Any] = None
    
    # Gauges
    active_connections: Optional[Any] = None
    queue_size: Optional[Any] = None
    indexed_vectors: Optional[Any] = None
    
    # Histograms
    gateway_latency_seconds: Optional[Any] = None
    structural_detector_latency_seconds: Optional[Any] = None
    vector_search_latency_seconds: Optional[Any] = None
    ml_classifier_latency_seconds: Optional[Any] = None
    consensus_latency_seconds: Optional[Any] = None
    llm_call_latency_seconds: Optional[Any] = None
    embedding_latency_seconds: Optional[Any] = None
    
    # Summary
    request_size_bytes: Optional[Any] = None
    response_size_bytes: Optional[Any] = None


def create_metrics_registry(settings: Settings = default_settings) -> MetricsRegistry:
    """Create and initialize metrics registry."""
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not available, metrics disabled")
        return MetricsRegistry()
    
    registry = CollectorRegistry()
    m = MetricsRegistry(_registry=registry)
    
    # Labels for multi-dimensional metrics
    gateway_labels = ["action", "application", "technique"]
    detector_labels = ["attack_type", "severity"]
    provider_labels = ["provider", "model", "status"]
    consensus_labels = ["decision", "consensus_reached"]
    
    # --- Counters ---
    m.gateway_requests_total = Counter(
        "ares_gateway_requests_total",
        "Total gateway enforcement requests",
        ["action"],
        registry=registry,
    )
    
    m.gateway_blocks_total = Counter(
        "ares_gateway_blocks_total",
        "Total blocked requests",
        ["application", "technique"],
        registry=registry,
    )
    
    m.gateway_sanitizes_total = Counter(
        "ares_gateway_sanitizes_total",
        "Total sanitized requests",
        ["application", "technique"],
        registry=registry,
    )
    
    m.gateway_allows_total = Counter(
        "ares_gateway_allows_total",
        "Total allowed requests",
        ["application"],
        registry=registry,
    )
    
    m.structural_detections_total = Counter(
        "ares_structural_detections_total",
        "Total structural pattern detections",
        ["attack_type", "severity"],
        registry=registry,
    )
    
    m.structural_blocks_total = Counter(
        "ares_structural_blocks_total",
        "Total structural blocks",
        ["attack_type"],
        registry=registry,
    )
    
    m.vector_searches_total = Counter(
        "ares_vector_searches_total",
        "Total vector similarity searches",
        registry=registry,
    )
    
    m.vector_blocks_total = Counter(
        "ares_vector_blocks_total",
        "Total vector-based blocks",
        ["technique"],
        registry=registry,
    )
    
    m.ml_classifications_total = Counter(
        "ares_ml_classifications_total",
        "Total ML classifier predictions",
        ["category", "confidence_bucket"],
        registry=registry,
    )
    
    m.consensus_decisions_total = Counter(
        "ares_consensus_decisions_total",
        "Total consensus decisions",
        ["decision", "consensus_reached"],
        registry=registry,
    )
    
    m.consensus_conflicts_total = Counter(
        "ares_consensus_conflicts_total",
        "Total consensus conflicts requiring escalation",
        registry=registry,
    )
    
    m.audit_events_total = Counter(
        "ares_audit_events_total",
        "Total audit events recorded",
        ["status"],
        registry=registry,
    )
    
    m.audit_failures_total = Counter(
        "ares_audit_failures_total",
        "Total audit logging failures",
        registry=registry,
    )
    
    m.llm_calls_total = Counter(
        "ares_llm_calls_total",
        "Total LLM provider calls",
        ["provider", "model", "status"],
        registry=registry,
    )
    
    m.llm_call_failures_total = Counter(
        "ares_llm_call_failures_total",
        "Total LLM call failures",
        ["provider", "error_type"],
        registry=registry,
    )
    
    m.llm_fallback_total = Counter(
        "ares_llm_fallback_total",
        "Total LLM provider fallbacks",
        ["from_provider", "to_provider"],
        registry=registry,
    )
    
    m.threat_intel_fetched_total = Counter(
        "ares_threat_intel_fetched_total",
        "Total threat intelligence samples fetched",
        ["source"],
        registry=registry,
    )
    
    m.threat_intel_indexed_total = Counter(
        "ares_threat_intel_indexed_total",
        "Total threat intel samples indexed",
        registry=registry,
    )
    
    m.threat_intel_rejected_total = Counter(
        "ares_threat_intel_rejected_total",
        "Total threat intel samples rejected",
        ["reason"],
        registry=registry,
    )
    
    m.moderation_submitted_total = Counter(
        "ares_moderation_submitted_total",
        "Total samples submitted for moderation",
        registry=registry,
    )
    
    m.moderation_approved_total = Counter(
        "ares_moderation_approved_total",
        "Total moderation samples approved",
        registry=registry,
    )
    
    m.moderation_rejected_total = Counter(
        "ares_moderation_rejected_total",
        "Total moderation samples rejected",
        ["reason"],
        registry=registry,
    )
    
    # --- Gauges ---
    m.active_connections = Gauge(
        "ares_active_connections",
        "Number of active connections",
        registry=registry,
    )
    
    m.queue_size = Gauge(
        "ares_queue_size",
        "Size of processing queues",
        ["queue_name"],
        registry=registry,
    )
    
    m.indexed_vectors = Gauge(
        "ares_indexed_vectors",
        "Number of indexed attack vectors in Qdrant",
        registry=registry,
    )
    
    # --- Histograms ---
    m.gateway_latency_seconds = Histogram(
        "ares_gateway_latency_seconds",
        "Gateway enforcement latency",
        ["action"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        registry=registry,
    )
    
    m.structural_detector_latency_seconds = Histogram(
        "ares_structural_detector_latency_seconds",
        "Structural detector latency",
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        registry=registry,
    )
    
    m.vector_search_latency_seconds = Histogram(
        "ares_vector_search_latency_seconds",
        "Vector similarity search latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=registry,
    )
    
    m.ml_classifier_latency_seconds = Histogram(
        "ares_ml_classifier_latency_seconds",
        "ML classifier inference latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=registry,
    )
    
    m.consensus_latency_seconds = Histogram(
        "ares_consensus_latency_seconds",
        "Consensus engine latency",
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
        registry=registry,
    )
    
    m.llm_call_latency_seconds = Histogram(
        "ares_llm_call_latency_seconds",
        "LLM provider call latency",
        ["provider"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        registry=registry,
    )
    
    m.embedding_latency_seconds = Histogram(
        "ares_embedding_latency_seconds",
        "Text embedding generation latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        registry=registry,
    )
    
    # --- Summaries ---
    m.request_size_bytes = Summary(
        "ares_request_size_bytes",
        "HTTP request body size in bytes",
        registry=registry,
    )
    
    m.response_size_bytes = Summary(
        "ares_response_size_bytes",
        "HTTP response body size in bytes",
        registry=registry,
    )
    
    return m


# Global metrics registry
_metrics: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get or create the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = create_metrics_registry()
    return _metrics


def get_metrics_output() -> bytes:
    """Get Prometheus metrics output for /metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n"
    from prometheus_client import generate_latest
    return generate_latest()


# --- Structured Logging ---

class StructuredLogger:
    """Structured logger with consistent fields."""
    
    def __init__(self, name: str, extra_fields: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.extra_fields = extra_fields or {}
    
    def _log(self, level: int, message: str, **kwargs):
        extra = {**self.extra_fields, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional bound fields."""
        new_extra = {**self.extra_fields, **kwargs}
        return StructuredLogger(self.logger.name, new_extra)


def get_structured_logger(name: str, **extra_fields) -> StructuredLogger:
    """Create a structured logger with bound fields."""
    return StructuredLogger(name, extra_fields)


# --- Timing Decorators ---

def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution and record to histogram."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            histogram = getattr(metrics, metric_name, None)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                if histogram:
                    duration = time.perf_counter() - start
                    if labels:
                        histogram.labels(**labels).observe(duration)
                    else:
                        histogram.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics()
            histogram = getattr(metrics, metric_name, None)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if histogram:
                    duration = time.perf_counter() - start
                    if labels:
                        histogram.labels(**labels).observe(duration)
                    else:
                        histogram.observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# --- Metrics Middleware ---

class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""
    
    def __init__(self, app, settings: Optional[Settings] = None):
        self.app = app
        self.settings = settings or default_settings
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        metrics = get_metrics()
        start_time = time.perf_counter()
        
        # Track request size
        body = b""
        async def receive_wrapper():
            nonlocal body
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
            return message
        
        # Capture response
        status_code = 200
        response_body = b""
        
        async def send_wrapper(message):
            nonlocal status_code, response_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)
        
        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            duration = time.perf_counter() - start_time
            path = scope.get("path", "unknown")
            method = scope.get("method", "UNKNOWN")
            
            # Record metrics
            if metrics.gateway_latency_seconds:
                metrics.gateway_latency_seconds.labels(action=path).observe(duration)
            
            if metrics.request_size_bytes:
                metrics.request_size_bytes.observe(len(body))
            
            if metrics.response_size_bytes:
                metrics.response_size_bytes.observe(len(response_body))


# --- Convenience Functions ---

def record_gateway_decision(action: str, application: str, technique: str = "unknown"):
    """Record a gateway enforcement decision."""
    metrics = get_metrics()
    if metrics.gateway_requests_total:
        metrics.gateway_requests_total.labels(action=action).inc()
    if action == "BLOCK" and metrics.gateway_blocks_total:
        metrics.gateway_blocks_total.labels(application=application, technique=technique).inc()
    elif action == "SANITIZE" and metrics.gateway_sanitizes_total:
        metrics.gateway_sanitizes_total.labels(application=application, technique=technique).inc()
    elif action == "ALLOW" and metrics.gateway_allows_total:
        metrics.gateway_allows_total.labels(application=application).inc()


def record_structural_detection(attack_type: str, severity: str, blocked: bool = False):
    """Record a structural pattern detection."""
    metrics = get_metrics()
    if metrics.structural_detections_total:
        metrics.structural_detections_total.labels(attack_type=attack_type, severity=severity).inc()
    if blocked and metrics.structural_blocks_total:
        metrics.structural_blocks_total.labels(attack_type=attack_type).inc()


def record_vector_search(blocked: bool = False, technique: str = "unknown"):
    """Record a vector similarity search."""
    metrics = get_metrics()
    if metrics.vector_searches_total:
        metrics.vector_searches_total.inc()
    if blocked and metrics.vector_blocks_total:
        metrics.vector_blocks_total.labels(technique=technique).inc()


def record_ml_classification(category: str, confidence: float):
    """Record an ML classifier prediction."""
    metrics = get_metrics()
    if metrics.ml_classifications_total:
        bucket = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
        metrics.ml_classifications_total.labels(category=category, confidence_bucket=bucket).inc()


def record_consensus_decision(decision: str, consensus_reached: bool):
    """Record a consensus engine decision."""
    metrics = get_metrics()
    if metrics.consensus_decisions_total:
        metrics.consensus_decisions_total.labels(
            decision=decision, 
            consensus_reached="true" if consensus_reached else "false"
        ).inc()
    if not consensus_reached and metrics.consensus_conflicts_total:
        metrics.consensus_conflicts_total.inc()


def record_audit_event(success: bool = True):
    """Record an audit logging event."""
    metrics = get_metrics()
    if metrics.audit_events_total:
        metrics.audit_events_total.labels(status="success" if success else "failure").inc()
    if not success and metrics.audit_failures_total:
        metrics.audit_failures_total.inc()


def record_llm_call(provider: str, model: str, success: bool, latency: float):
    """Record an LLM provider call."""
    metrics = get_metrics()
    if metrics.llm_calls_total:
        metrics.llm_calls_total.labels(
            provider=provider, 
            model=model, 
            status="success" if success else "failure"
        ).inc()
    if not success and metrics.llm_call_failures_total:
        metrics.llm_call_failures_total.labels(provider=provider, error_type="unknown").inc()
    if metrics.llm_call_latency_seconds:
        metrics.llm_call_latency_seconds.labels(provider=provider).observe(latency)


def record_llm_fallback(from_provider: str, to_provider: str):
    """Record an LLM provider fallback."""
    metrics = get_metrics()
    if metrics.llm_fallback_total:
        metrics.llm_fallback_total.labels(
            from_provider=from_provider,
            to_provider=to_provider
        ).inc()


def record_threat_intel_fetched(source: str, count: int = 1):
    """Record threat intelligence fetch."""
    metrics = get_metrics()
    if metrics.threat_intel_fetched_total:
        metrics.threat_intel_fetched_total.labels(source=source).inc(count)


def record_threat_intel_indexed(count: int = 1):
    """Record threat intelligence indexed."""
    metrics = get_metrics()
    if metrics.threat_intel_indexed_total:
        metrics.threat_intel_indexed_total.inc(count)


def record_threat_intel_rejected(reason: str):
    """Record threat intelligence rejected."""
    metrics = get_metrics()
    if metrics.threat_intel_rejected_total:
        metrics.threat_intel_rejected_total.labels(reason=reason).inc()


def record_moderation_submitted():
    """Record moderation sample submitted."""
    metrics = get_metrics()
    if metrics.moderation_submitted_total:
        metrics.moderation_submitted_total.inc()


def record_moderation_approved():
    """Record moderation sample approved."""
    metrics = get_metrics()
    if metrics.moderation_approved_total:
        metrics.moderation_approved_total.inc()


def record_moderation_rejected(reason: str):
    """Record moderation sample rejected."""
    metrics = get_metrics()
    if metrics.moderation_rejected_total:
        metrics.moderation_rejected_total.labels(reason=reason).inc()


def set_indexed_vectors(count: int):
    """Set the number of indexed vectors gauge."""
    metrics = get_metrics()
    if metrics.indexed_vectors:
        metrics.indexed_vectors.set(count)


def set_queue_size(queue_name: str, size: int):
    """Set a queue size gauge."""
    metrics = get_metrics()
    if metrics.queue_size:
        metrics.queue_size.labels(queue_name=queue_name).set(size)


# Initialize metrics on import
_metrics = create_metrics_registry()