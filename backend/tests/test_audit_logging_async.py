"""
Unit tests for non-blocking async audit logging offload in RuntimeEnforcer.
Verifies that database transactions do not block the asyncio event loop or user requests.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ares.api.schemas import EnforceRequest, EnforceActionEnum
from ares.config import settings
from ares.gateway.enforcer import RuntimeEnforcer
from ares.vectordb.store import QdrantStore


@pytest.fixture
def mock_store():
    store = MagicMock(spec=QdrantStore)
    store.search_similar_attacks = AsyncMock(return_value=[])
    store.embedder = MagicMock()
    store.embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
    store.embedder.embed_batch = AsyncMock(return_value=[[0.1] * 768] * 10)
    return store


@pytest.mark.asyncio
async def test_audit_logging_is_non_blocking_under_db_latency(mock_store):
    """
    Verify that even if the database transaction takes 200ms of simulated latency,
    the enforce() call returns immediately (<50ms) without blocking the event loop.
    """
    enforcer = RuntimeEnforcer(settings=settings, store=mock_store)

    db_persist_completed = False

    def slow_persist_worker(decision, request, *args, **kwargs):
        nonlocal db_persist_completed
        time.sleep(0.2)  # Simulate 200ms database I/O latency
        db_persist_completed = True

    # Patch the worker method with slow persistence
    with patch.object(enforcer, "_persist_audit_event_sync", side_effect=slow_persist_worker):
        req = EnforceRequest(
            prompt="Hello, this is a clean query for latency testing.",
            application_name="PerfTester",
        )

        start_time = time.perf_counter()
        decision = await enforcer.enforce(req)
        elapsed_time = time.perf_counter() - start_time

        # enforce() must return immediately, NOT wait for 200ms DB write
        assert elapsed_time < 0.1, f"enforce() was blocked by DB logging! Elapsed: {elapsed_time:.3f}s"
        assert decision.action == EnforceActionEnum.allow

        # At the immediate moment of return, DB write is still running in background
        assert db_persist_completed is False

        # Allow background thread to finish
        await asyncio.sleep(0.25)
        assert db_persist_completed is True


@pytest.mark.asyncio
async def test_audit_logging_fallback_in_sync_context(mock_store):
    """
    Verify that when called in a synchronous thread with no active event loop,
    _record_audit_event falls back safely to synchronous execution without error.
    """
    enforcer = RuntimeEnforcer(settings=settings, store=mock_store)
    called = False

    def mock_persist(decision, request, *args, **kwargs):
        nonlocal called
        called = True

    with patch.object(enforcer, "_persist_audit_event_sync", side_effect=mock_persist):
        req = EnforceRequest(prompt="test", application_name="test")
        from ares.api.schemas import EnforceResponse, SeverityEnum
        from datetime import datetime, timezone

        resp = EnforceResponse(
            action=EnforceActionEnum.allow,
            risk_score=0,
            severity=SeverityEnum.safe,
            matched_technique=None,
            vector_similarity=0.0,
            sanitized_prompt=None,
            explanation="test",
            latency_ms=1.0,
            correlation_id="test-123",
            timestamp=datetime.now(timezone.utc),
        )

        # Force get_running_loop to raise RuntimeError
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            enforcer._record_audit_event(resp, req)
            assert called is True
