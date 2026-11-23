"""Tests for CircuitBreaker secret/PII log scrubbing and error sanitization."""

from __future__ import annotations

import logging

from ares.security.circuit_breaker import CircuitBreaker
from ares.security.error_sanitizer import sanitize_error_message


def test_circuit_breaker_redacts_secrets_and_pii_from_warning_logs(caplog):
    breaker = CircuitBreaker("upstream-llm", failure_threshold=3, recovery_timeout_seconds=30.0)
    raw_secret_err = RuntimeError(
        "Request failed for sk-proj-supersecret1234567890, gsk_live_secret_token_9876543210, "
        "AIzaSyDummyGoogleKey12345678901234567890, ares_tnt_org_acme_secret42, "
        "Authorization: Bearer tok_super_secret_9988, "
        "db=postgresql://postgres:topsecret@127.0.0.1:5432/ares, ssn=123-45-6789"
    )

    with caplog.at_level(logging.WARNING, logger="ares.security.circuit_breaker"):
        breaker.record_failure(raw_secret_err)

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Circuit breaker 'upstream-llm' recorded failure (1/3): RuntimeError:" in joined
    assert "sk-proj-supersecret1234567890" not in joined
    assert "gsk_live_secret_token_9876543210" not in joined
    assert "AIzaSyDummyGoogleKey12345678901234567890" not in joined
    assert "ares_tnt_org_acme_secret42" not in joined
    assert "tok_super_secret_9988" not in joined
    assert "postgresql://postgres:topsecret@127.0.0.1:5432/ares" not in joined
    assert "123-45-6789" not in joined
    assert "[REDACTED_API_KEY]" in joined
    assert "Bearer [REDACTED_TOKEN]" in joined
    assert "[REDACTED_DB_URI]" in joined
    assert "[REDACTED_SSN]" in joined


def test_sanitize_error_message_truncates_and_preserves_safe_text():
    cleaned = sanitize_error_message("Connection timeout talking to upstream host", max_len=24)
    assert len(cleaned) <= 24
    assert cleaned.endswith("...")
