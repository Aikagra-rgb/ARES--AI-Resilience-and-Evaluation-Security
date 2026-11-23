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


def test_memory_rate_limiter_lazy_cleanup_and_max_keys_bound():
    """Verify MemoryRateLimiter evicts expired keys lazily and enforces max_keys cardinality."""
    limiter = MemoryRateLimiter(max_keys=3, cleanup_interval_seconds=0.01, max_idle_seconds=0)
    limiter.check("k1", limit=5, window_seconds=10)
    limiter.check("k2", limit=5, window_seconds=10)
    limiter.check("k3", limit=5, window_seconds=10)
    time.sleep(0.05)

    # Triggers lazy cleanup & max_keys enforcement before adding k4
    limiter.check("k4", limit=5, window_seconds=10)
    assert len(limiter._windows) <= 3
    assert "k4" in limiter._windows

    # Now test hard cardinality bound when all keys are fresh (large max_idle_seconds)
    bounded = MemoryRateLimiter(max_keys=2, cleanup_interval_seconds=3600.0, max_idle_seconds=3600)
    bounded.check("fresh_1", limit=5, window_seconds=60)
    bounded.check("fresh_2", limit=5, window_seconds=60)
    bounded.check("fresh_3", limit=5, window_seconds=60)
    assert len(bounded._windows) == 2
    assert "fresh_3" in bounded._windows


class _FakeRedisPipeline:
    def __init__(self, backend):
        self._backend = backend
        self._ops = []

    def zremrangebyscore(self, key, min_score, max_score):
        self._ops.append(("zrem", key, float(min_score), float(max_score)))
        return self

    def zcard(self, key):
        self._ops.append(("zcard", key))
        return self

    def zrange(self, key, start, stop, withscores=False):
        self._ops.append(("zrange", key, start, stop, withscores))
        return self

    def zadd(self, key, mapping):
        self._ops.append(("zadd", key, dict(mapping)))
        return self

    def pexpire(self, key, ttl_ms):
        self._ops.append(("pexpire", key, ttl_ms))
        return self

    def execute(self):
        if self._backend.fail_next:
            raise ConnectionError("Simulated Redis outage")
        results = []
        for op in self._ops:
            kind = op[0]
            key = op[1]
            zset = self._backend.sets.setdefault(key, [])
            if kind == "zrem":
                _, _, min_s, max_s = op
                zset[:] = [(m, s) for (m, s) in zset if not (min_s <= s <= max_s)]
                results.append(len(zset))
            elif kind == "zcard":
                results.append(len(zset))
            elif kind == "zrange":
                _, _, start, stop, withscores = op
                ordered = sorted(zset, key=lambda x: x[1])
                sliced = ordered[start : (None if stop == -1 else stop + 1)]
                results.append(sliced if withscores else [m for m, _ in sliced])
            elif kind == "zadd":
                for m, s in op[2].items():
                    zset.append((m, float(s)))
                results.append(len(op[2]))
            elif kind == "pexpire":
                results.append(True)
        return results


class _FakeRedis:
    def __init__(self):
        self.sets = {}
        self.fail_next = False

    def pipeline(self):
        return _FakeRedisPipeline(self)


def test_redis_rate_limiter_sliding_window_and_fallback():
    from ares.security.rate_limiter import RedisRateLimiter, create_rate_limiter

    assert isinstance(create_rate_limiter(None), MemoryRateLimiter)
    assert isinstance(create_rate_limiter("redis://127.0.0.1:6379/0"), RedisRateLimiter)

    fake_redis = _FakeRedis()
    limiter = RedisRateLimiter(redis_client=fake_redis)
    assert limiter.backend_name == "redis"

    r1 = limiter.check("tenant:t1", limit=2, window_seconds=10)
    assert r1.allowed is True
    assert r1.remaining == 1

    r2 = limiter.check("tenant:t1", limit=2, window_seconds=10)
    assert r2.allowed is True
    assert r2.remaining == 0

    r3 = limiter.check("tenant:t1", limit=2, window_seconds=10)
    assert r3.allowed is False
    assert r3.remaining == 0
    assert r3.reset_after >= 1

    # Simulate Redis failure -> automatic failover to MemoryRateLimiter
    fake_redis.fail_next = True
    r4 = limiter.check("tenant:t1", limit=2, window_seconds=10)
    assert r4.allowed is True
    assert limiter.backend_name == "memory_fallback"

