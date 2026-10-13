"""
Tests for ARES Structured JSON Logging and Contextual Tracing.

Verifies:
1. Output format complies with newline-delimited JSON schema.
2. Automatic context enrichment with correlation_id and organization_id.
3. Exception formatting into structured JSON.
4. Custom extra attributes extraction.
"""

import json
import logging
import pytest

from ares.observability.logging import (
    StructuredJsonFormatter,
    get_correlation_id,
    set_correlation_id,
    get_organization_id,
    set_organization_id,
)


def test_structured_json_formatter_basic():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="ares.gateway.test",
        level=logging.INFO,
        pathname="gateway/test.py",
        lineno=42,
        msg="Inbound request evaluated successfully.",
        args=(),
        exc_info=None,
        func="test_func",
    )

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "INFO"
    assert data["logger"] == "ares.gateway.test"
    assert data["message"] == "Inbound request evaluated successfully."
    assert data["function"] == "test_func"
    assert data["line"] == 42
    assert "timestamp" in data


def test_structured_json_formatter_context_enrichment():
    formatter = StructuredJsonFormatter()

    set_correlation_id("ares-corr-12345")
    set_organization_id("org-enterprise-corp")

    try:
        record = logging.LogRecord(
            name="ares.security.enforcer",
            level=logging.WARNING,
            pathname="gateway/enforcer.py",
            lineno=100,
            msg="Adversarial vector detected above threshold.",
            args=(),
            exc_info=None,
            func="enforce",
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["correlation_id"] == "ares-corr-12345"
        assert data["organization_id"] == "org-enterprise-corp"
    finally:
        set_correlation_id(None)
        set_organization_id(None)


def test_structured_json_formatter_with_exception():
    formatter = StructuredJsonFormatter()

    try:
        raise ValueError("Simulated external connection drop")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="ares.security.circuit_breaker",
            level=logging.ERROR,
            pathname="security/circuit_breaker.py",
            lineno=80,
            msg="Circuit breaker tripped to OPEN state.",
            args=(),
            exc_info=sys.exc_info(),
            func="execute",
        )

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert "exception" in data
    assert "ValueError: Simulated external connection drop" in data["exception"]


def test_structured_json_formatter_with_extra_fields():
    formatter = StructuredJsonFormatter()

    record = logging.LogRecord(
        name="ares.metrics",
        level=logging.INFO,
        pathname="observability/metrics.py",
        lineno=50,
        msg="Recorded gateway decision metric.",
        args=(),
        exc_info=None,
        func="record_metric",
    )
    record.decision = "BLOCK"
    record.latency_ms = 4.2
    record.risk_score = 95

    output = formatter.format(record)
    data = json.loads(output)

    assert "extra" in data
    assert data["extra"]["decision"] == "BLOCK"
    assert data["extra"]["latency_ms"] == 4.2
    assert data["extra"]["risk_score"] == 95
