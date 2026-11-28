"""Tests for explicit production fail policy (HMAC modes and fail-closed Settings)."""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ares.config import Settings
from ares.security.signing import HmacVerificationMiddleware, HmacVerifier


def _build_hmac_app(enforce_mode: str, secret: str) -> FastAPI:
    test_app = FastAPI()
    verifier = HmacVerifier(default_secret=secret, max_clock_skew_seconds=300)
    test_app.add_middleware(HmacVerificationMiddleware, verifier=verifier, enforce_mode=enforce_mode)

    @test_app.post("/api/gateway/enforce")
    def _enforce():
        return {"ok": True}

    @test_app.get("/api/v1/clients")
    def _clients():
        return {"items": []}

    @test_app.get("/v1/models")
    def _models():
        return {"object": "list", "data": []}

    @test_app.get("/healthz")
    def _healthz():
        return {"status": "ok"}

    return test_app


def _sign(method: str, path: str, body: str, secret: str) -> dict[str, str]:
    now = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest().lower() if body else ""
    canonical = f"{method.upper()}\n{path}\n\n{body_sha}\n{now}\n{nonce}"
    sig = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest().lower()
    return {
        "X-Ares-Timestamp": now,
        "X-Ares-Nonce": nonce,
        "X-Ares-Signature": sig,
        "X-Ares-Key-Id": "ares-web",
    }


def test_hmac_enforcing_critical_protects_critical_routes_and_allows_openai_sdk_path():
    secret = "prod-hmac-secret-key-at-least-24-chars"
    client = TestClient(_build_hmac_app("enforcing_critical", secret))

    # Critical gateway & client control routes MUST reject unsigned requests
    enf_unsigned = client.post("/api/gateway/enforce", json={"prompt": "hi"})
    assert enf_unsigned.status_code == 401
    assert enf_unsigned.json()["code"] == "HMAC_AUTH_REQUIRED"

    cli_unsigned = client.get("/api/v1/clients")
    assert cli_unsigned.status_code == 401
    assert cli_unsigned.json()["code"] == "HMAC_AUTH_REQUIRED"

    # Standard OpenAI proxy path permits API-key-only SDK callers when unsigned
    assert client.get("/v1/models").status_code == 200
    assert client.get("/healthz").status_code == 200

    # Valid signed critical request succeeds
    body = '{"prompt":"hi"}'
    hdrs = _sign("POST", "/api/gateway/enforce", body, secret)
    enf_signed = client.post(
        "/api/gateway/enforce",
        content=body,
        headers={**hdrs, "Content-Type": "application/json"},
    )
    assert enf_signed.status_code == 200

    # Tampered signature on ANY non-exempt route (even /v1/models) is rejected
    bad_hdrs = _sign("GET", "/v1/models", "", secret)
    bad_hdrs["X-Ares-Signature"] = "f" * 64
    tampered = client.get("/v1/models", headers=bad_hdrs)
    assert tampered.status_code == 403
    assert tampered.json()["code"] == "HMAC_SIGNATURE_INVALID"


def test_hmac_full_enforcing_and_invalid_mode_rejection():
    secret = "prod-hmac-secret-key-at-least-24-chars"
    client = TestClient(_build_hmac_app("enforcing", secret))

    # Even /v1/models requires HMAC under full 'enforcing' mode, while /healthz stays exempt
    assert client.get("/v1/models").status_code == 401
    assert client.get("/healthz").status_code == 200

    with pytest.raises(ValueError, match="Invalid HMAC"):
        HmacVerificationMiddleware(FastAPI(), verifier=HmacVerifier(), enforce_mode="invalid-mode")


def test_settings_production_fail_closed_defaults_and_rejections():
    dev_cfg = Settings(DATABASE_URL="sqlite:///test.db", ENVIRONMENT="test")
    assert dev_cfg.hmac_enforce_mode == "permissive"
    assert dev_cfg.gateway_fail_closed_on_outage is False

    prod_kwargs = {
        "DATABASE_URL": "postgresql+psycopg://ares:pass@postgres:5432/ares",
        "ENVIRONMENT": "production",
        "ARES_AUTH_ENABLED": True,
        "ALLOW_DEV_IDENTITY": False,
        "PROMPT_FINGERPRINT_SECRET": "fp-" + "x" * 24,
        "EXECUTION_PAYLOAD_ENCRYPTION_KEY": "fernet-" + "y" * 24,
        "ARES_API_KEY": "ares-" + "z" * 24,
        "HMAC_SECRET": "hmac-" + "k" * 24,
        "GATEWAY_CANARY_TOKEN": "CANARY_PROD_01",
    }
    prod_cfg = Settings(**prod_kwargs)
    assert prod_cfg.hmac_enforce_mode == "enforcing_critical"
    assert prod_cfg.gateway_fail_closed_on_outage is True

    with pytest.raises(ValueError, match="HMAC_ENFORCE_MODE"):
        Settings(**{**prod_kwargs, "HMAC_ENFORCE_MODE": "disabled"})

    with pytest.raises(ValueError, match="HMAC_SECRET"):
        Settings(**{**prod_kwargs, "HMAC_SECRET": "ares-dev-secret-key-42"})
