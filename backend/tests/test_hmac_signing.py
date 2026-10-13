"""
Unit & Integration Tests for ARES Server-Side HMAC-SHA256 Verification.
Validates compatibility with .NET FastApiAresApiClient signing protocol.
"""

import hashlib
import hmac
import time
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app, hmac_verifier
from ares.security.signing import HmacVerifier


@pytest.fixture
def test_secret():
    return "test-signing-secret-key-1234567890"


@pytest.fixture
def verifier(test_secret):
    return HmacVerifier(
        default_secret=test_secret,
        max_clock_skew_seconds=300,
    )


def compute_test_signature(
    method: str,
    path: str,
    query: str,
    body: str,
    timestamp: str,
    nonce: str,
    secret: str,
) -> str:
    """Canonical request computation mirroring .NET FastApiAresApiClient.cs."""
    body_sha256 = (
        hashlib.sha256(body.encode("utf-8")).hexdigest().lower()
        if body
        else ""
    )
    canonical = f"{method.upper()}\n{path}\n{query.lstrip('?')}\n{body_sha256}\n{timestamp}\n{nonce}"
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest().lower()


def test_hmac_verifier_valid_signature(verifier, test_secret):
    now = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body = '{"prompt": "hello world"}'
    sig = compute_test_signature("POST", "/api/gateway/enforce", "", body, now, nonce, test_secret)

    is_valid, reason = verifier.verify(
        method="POST",
        path="/api/gateway/enforce",
        query="",
        body=body,
        timestamp_str=now,
        nonce=nonce,
        signature=sig,
    )
    assert is_valid is True
    assert reason == "Valid"


def test_hmac_verifier_tampered_body_rejected(verifier, test_secret):
    now = str(int(time.time()))
    nonce = uuid.uuid4().hex
    original_body = '{"prompt": "hello world"}'
    sig = compute_test_signature("POST", "/api/gateway/enforce", "", original_body, now, nonce, test_secret)

    # Tampered body
    tampered_body = '{"prompt": "hello world - TAMPERED"}'
    is_valid, reason = verifier.verify(
        method="POST",
        path="/api/gateway/enforce",
        query="",
        body=tampered_body,
        timestamp_str=now,
        nonce=nonce,
        signature=sig,
    )
    assert is_valid is False
    assert "mismatch" in reason.lower()


def test_hmac_verifier_replay_attack_rejected(verifier, test_secret):
    now = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body = '{"prompt": "hello"}'
    sig = compute_test_signature("POST", "/api/gateway/enforce", "", body, now, nonce, test_secret)

    # First attempt: valid
    is_valid, _ = verifier.verify(
        method="POST",
        path="/api/gateway/enforce",
        query="",
        body=body,
        timestamp_str=now,
        nonce=nonce,
        signature=sig,
    )
    assert is_valid is True

    # Second attempt (replay of same nonce): rejected
    is_valid_replay, reason = verifier.verify(
        method="POST",
        path="/api/gateway/enforce",
        query="",
        body=body,
        timestamp_str=now,
        nonce=nonce,
        signature=sig,
    )
    assert is_valid_replay is False
    assert "replay" in reason.lower()


def test_hmac_verifier_clock_skew_expired(verifier, test_secret):
    # Timestamp 10 minutes in the past (> 300s)
    old_time = str(int(time.time()) - 600)
    nonce = uuid.uuid4().hex
    body = ""
    sig = compute_test_signature("GET", "/api/models", "", body, old_time, nonce, test_secret)

    is_valid, reason = verifier.verify(
        method="GET",
        path="/api/models",
        query="",
        body=body,
        timestamp_str=old_time,
        nonce=nonce,
        signature=sig,
    )
    assert is_valid is False
    assert "clock skew" in reason.lower()


def test_hmac_middleware_integration_with_fastapi():
    """Test full HTTP middleware lifecycle using TestClient."""
    secret = hmac_verifier.default_secret
    client = TestClient(app)

    # 1. Health check bypasses HMAC
    health_resp = client.get("/healthz")
    assert health_resp.status_code == 200

    # 2. Signed request to /api/models with valid HMAC signature
    now = str(int(time.time()))
    nonce = uuid.uuid4().hex
    sig = compute_test_signature("GET", "/api/models", "", "", now, nonce, secret)

    headers = {
        "X-Ares-Timestamp": now,
        "X-Ares-Nonce": nonce,
        "X-Ares-Signature": sig,
        "X-Ares-Key-Id": "ares-web",
    }
    resp = client.get("/api/models", headers=headers)
    assert resp.status_code == 200

    # 3. Request with tampered signature -> 403 Forbidden
    tampered_headers = dict(headers)
    tampered_headers["X-Ares-Signature"] = "0000000000000000000000000000000000000000000000000000000000000000"
    tampered_resp = client.get("/api/models", headers=tampered_headers)
    assert tampered_resp.status_code in (401, 403)
    assert "HMAC_SIGNATURE_INVALID" in tampered_resp.json()["code"]
