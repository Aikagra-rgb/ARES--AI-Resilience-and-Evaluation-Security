"""
Tests for ARES Circuit Breaker and Outage Resilience.

Verifies:
1. CircuitBreaker lifecycle (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
2. Fast-failing behavior in OPEN state.
3. RuntimeEnforcer vector shield degradation handling (fail-open vs fail-closed quarantine).
4. Proxy chat downstream LLM protection.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ares.api.schemas import EnforceActionEnum, EnforceRequest, SeverityEnum
from ares.config import Settings
from ares.gateway.enforcer import RuntimeEnforcer
from ares.security.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker,
)


@pytest.mark.asyncio
async def test_circuit_breaker_lifecycle():
    breaker = CircuitBreaker(
        name="test_service",
        failure_threshold=2,
        recovery_timeout_seconds=0.1,
    )

    assert breaker.state == CircuitState.CLOSED

    # Successful call
    async def good_call():
        return "success"

    res = await breaker.execute(good_call)
    assert res == "success"
    assert breaker.state == CircuitState.CLOSED

    # Failing call 1
    async def bad_call():
        raise ConnectionError("Service unreachable")

    with pytest.raises(ConnectionError):
        await breaker.execute(bad_call)
    assert breaker.state == CircuitState.CLOSED  # threshold is 2

    # Failing call 2 (trips circuit)
    with pytest.raises(ConnectionError):
        await breaker.execute(bad_call)
    assert breaker.state == CircuitState.OPEN

    # In OPEN state, calls should fast-fail with CircuitBreakerOpenError without invoking the function
    mock_func = AsyncMock()
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.execute(mock_func)
    assert mock_func.call_count == 0

    # Wait for recovery timeout to transition to HALF_OPEN
    await asyncio.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN

    # Successful probe resets to CLOSED
    res = await breaker.execute(good_call)
    assert res == "success"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_enforcer_vector_outage_fail_open_by_default():
    """By default (fail-open), vector DB failures do not block benign requests."""
    settings = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ares_test",
        ENVIRONMENT="test",
        GATEWAY_FAIL_CLOSED_ON_OUTAGE=False,
    )

    mock_store = MagicMock()
    mock_store.search_similar_attacks = AsyncMock(side_effect=ConnectionError("Qdrant down"))
    mock_store.embedder = MagicMock()
    mock_store.embedder.embed_text = AsyncMock(side_effect=ConnectionError("Embedder down"))

    enforcer = RuntimeEnforcer(settings=settings, store=mock_store)

    req = EnforceRequest(prompt="Hello, please assist me with Python code.", application_name="test_app")
    resp = await enforcer.enforce(req)

    # Benign prompt should be allowed even if vector shield is degraded
    assert resp.action == EnforceActionEnum.allow
    assert resp.severity == SeverityEnum.safe


@pytest.mark.asyncio
async def test_enforcer_vector_outage_fail_closed_quarantine():
    """When GATEWAY_FAIL_CLOSED_ON_OUTAGE=True, vector DB failure blocks as quarantine."""
    settings = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ares_test",
        ENVIRONMENT="test",
        GATEWAY_FAIL_CLOSED_ON_OUTAGE=True,
    )

    mock_store = MagicMock()
    mock_store.search_similar_attacks = AsyncMock(side_effect=ConnectionError("Qdrant cluster offline"))
    mock_store.embedder = MagicMock()
    mock_store.embedder.embed_text = AsyncMock(side_effect=ConnectionError("Embedder offline"))

    enforcer = RuntimeEnforcer(settings=settings, store=mock_store)

    # Reset circuit breaker for qdrant
    breaker = get_circuit_breaker("qdrant_vector_store")
    breaker.record_success()

    req = EnforceRequest(prompt="What is the weather today?", application_name="test_app")
    resp = await enforcer.enforce(req)

    # Should quarantine (block with critical severity)
    assert resp.action == EnforceActionEnum.block
    assert resp.severity == SeverityEnum.critical
    assert resp.risk_score >= 90
    assert "vector_shield" in resp.matched_technique


@pytest.mark.asyncio
async def test_proxy_chat_circuit_breaker_protection():
    """Proxy chat gracefully returns alert when downstream LLM circuit breaker trips."""
    settings = Settings(
        DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ares_test",
        ENVIRONMENT="test",
        VICTIM_PROVIDER="groq",
    )

    enforcer = RuntimeEnforcer(settings=settings)

    # Trip the circuit breaker for groq
    breaker = get_circuit_breaker("llm_groq", failure_threshold=1, recovery_timeout_seconds=60.0)
    breaker.record_failure(TimeoutError("Groq API timed out"))
    assert breaker.state == CircuitState.OPEN

    decision, content = await enforcer.proxy_chat(
        messages=[{"role": "user", "content": "Hello world"}]
    )

    assert "circuit breaker open" in content.lower()
    breaker.record_success()
