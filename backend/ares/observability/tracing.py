"""
ARES OpenTelemetry Distributed Tracing Instrumentation.

Provides tracing span management for gateway request lifecycles, vector DB queries,
and LLM downstream interactions with graceful no-op fallback when OpenTelemetry
is not installed or collector is unconfigured.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import logging
from typing import Any, AsyncIterator, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# Standard ARES Semantic Attributes
ATTR_ARES_LAYER = "ares.layer"
ATTR_ARES_DECISION = "ares.decision"
ATTR_ARES_SEVERITY = "ares.severity"
ATTR_ARES_RISK_SCORE = "ares.risk_score"
ATTR_ARES_CORRELATION_ID = "ares.correlation_id"
ATTR_ARES_TENANT_ID = "ares.tenant_id"
ATTR_ARES_TECHNIQUE = "ares.technique"

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    _tracer = trace.get_tracer("ares.security.gateway")
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    _tracer = None


class NoOpSpan:
    """Zero-overhead fallback span when OpenTelemetry is inactive."""

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "NoOpSpan":
        return self

    def record_exception(self, exception: Exception) -> "NoOpSpan":
        return self

    def set_status(self, status: Any, description: Optional[str] = None) -> "NoOpSpan":
        return self

    def end(self) -> None:
        pass


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """
    Synchronous context manager for tracing execution spans.
    """
    if not OPENTELEMETRY_AVAILABLE or _tracer is None:
        span = NoOpSpan()
        if attributes:
            span.set_attributes(attributes)
        yield span
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


@asynccontextmanager
async def async_trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Any]:
    """
    Asynchronous context manager for tracing execution spans.
    """
    if not OPENTELEMETRY_AVAILABLE or _tracer is None:
        span = NoOpSpan()
        if attributes:
            span.set_attributes(attributes)
        yield span
        return

    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
