"""
Unit & Integration Tests for ARES Gateway HTTP Rate Limiting.
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ares.security.rate_limiter import MemoryRateLimiter, RateLimitMiddleware


def test_memory_rate_limiter_basic():
    limiter = MemoryRateLimiter()
    key = "test_client_1"

    # Request 1 of 3: allowed, remaining 2
    res1 = limiter.check(key, limit=3, window_seconds=10)
    assert res1.allowed is True
    assert res1.remaining == 2
    assert res1.limit == 3

    # Request 2 of 3: allowed, remaining 1
    res2 = limiter.check(key, limit=3, window_seconds=10)
    assert res2.allowed is True
    assert res2.remaining == 1

    # Request 3 of 3: allowed, remaining 0
    res3 = limiter.check(key, limit=3, window_seconds=10)
    assert res3.allowed is True
    assert res3.remaining == 0

    # Request 4 of 3: blocked
    res4 = limiter.check(key, limit=3, window_seconds=10)
    assert res4.allowed is False
    assert res4.remaining == 0
    assert res4.reset_after > 0


def test_memory_rate_limiter_window_expiry():
    limiter = MemoryRateLimiter()
    key = "test_client_expiry"

    # Fill limit of 1 request per 1 second
    res1 = limiter.check(key, limit=1, window_seconds=1)
    assert res1.allowed is True

    res2 = limiter.check(key, limit=1, window_seconds=1)
    assert res2.allowed is False

    # Wait for window to expire
    time.sleep(1.1)

    res3 = limiter.check(key, limit=1, window_seconds=1)
    assert res3.allowed is True


def test_rate_limit_middleware_headers():
    client = TestClient(app)

    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert "X-RateLimit-Limit" in resp.headers
    assert "X-RateLimit-Remaining" in resp.headers
    assert "X-RateLimit-Reset" in resp.headers


def test_rate_limit_middleware_blocks_bursts():
    """Verify middleware returns 429 when custom limiter quota is exhausted."""
    from fastapi import FastAPI

    small_app = FastAPI()
    limiter = MemoryRateLimiter()
    small_app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        enabled=True,
        requests_per_window=2,
        window_seconds=10,
    )

    @small_app.get("/ping")
    def ping():
        return {"status": "pong"}

    client = TestClient(small_app)

    # 1. First request -> 200
    r1 = client.get("/ping")
    assert r1.status_code == 200
    assert r1.headers["X-RateLimit-Remaining"] == "1"

    # 2. Second request -> 200
    r2 = client.get("/ping")
    assert r2.status_code == 200
    assert r2.headers["X-RateLimit-Remaining"] == "0"

    # 3. Third request -> 429 Too Many Requests
    r3 = client.get("/ping")
    assert r3.status_code == 429
    assert r3.headers["Retry-After"] is not None
    assert r3.json()["code"] == "RATE_LIMIT_EXCEEDED"
