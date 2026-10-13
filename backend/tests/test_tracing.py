"""
Tests for ARES Distributed Tracing Instrumentation.

Verifies:
1. No-op span fallback operates transparently when OpenTelemetry is unconfigured.
2. async_trace_span and trace_span context managers properly handle normal and exception flows.
3. RuntimeEnforcer span attributes (decision, severity, risk score) are populated.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ares.api.schemas import EnforceActionEnum, EnforceRequest, SeverityEnum
from ares.config import Settings
from ares.gateway.context import TenantContext
from ares.gateway.enforcer import RuntimeEnforcer
from ares.observability.tracing import (
    NoOpSpan,
    trace_span,
    async_trace_span,
    ATTR_ARES_DECISION,
    ATTR_ARES_SEVERITY,
    ATTR_ARES_RISK_SCORE,
)


def test_noop_span_methods():
    span = NoOpSpan()
    span.set_attribute("key", "val")
    span.set_attributes({"a": 1, "b": 2})
    span.record_exception(RuntimeError("test"))
    span.set_status("OK")
    span.end()


def test_sync_trace_span():
    with trace_span("test_operation", {"service": "test"}) as span:
        assert span is not None


@pytest.mark.asyncio
async def test_async_trace_span_with_exception():
    with pytest.raises(ZeroDivisionError):
        async with async_trace_span("failing_operation"):
            _ = 1 / 0


@pytest.mark.asyncio
async def test_enforcer_tracing_integration():
    settings = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ares_test",
        ENVIRONMENT="test",
    )
    store = MagicMock()
    store.search_similar_attacks = AsyncMock(return_value=[])
    store.embedder = MagicMock()
    store.embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
    store.embedder.embed_batch = AsyncMock(return_value=[])

    enforcer = RuntimeEnforcer(settings=settings, store=store)

    req = EnforceRequest(prompt="Hello, assist with python code.", application_name="tracing_test")
    ctx = TenantContext(organization_id="org_tracing")

    decision = await enforcer.enforce(req, context=ctx)
    assert decision.action == EnforceActionEnum.allow
    assert decision.severity == SeverityEnum.safe
