"""
ARES HTTP Gateway Rate Limiting Engine.

Provides token bucket and sliding window rate limiting at the FastAPI application layer.
Supports both a thread-safe in-memory sliding window backend and a distributed Redis backend.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RateLimitResult:
    def __init__(self, allowed: bool, limit: int, remaining: int, reset_after: int):
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_after = reset_after


class MemoryRateLimiter:
    """
    Thread-safe in-memory sliding-window log rate limiter with lazy TTL eviction
    and bounded key cardinality.
    """

    def __init__(
        self,
        max_keys: int = 10_000,
        cleanup_interval_seconds: float = 60.0,
        max_idle_seconds: int = 3600,
    ):
        self._windows: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self._max_keys = max(1, int(max_keys))
        self._cleanup_interval_seconds = float(cleanup_interval_seconds)
        self._max_idle_seconds = int(max_idle_seconds)
        self._last_cleanup = time.time()

    def _cleanup_locked(self, now: float, max_idle_seconds: int) -> None:
        cutoff = now - max_idle_seconds
        stale_keys = [
            k for k, timestamps in self._windows.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for k in stale_keys:
            self._windows.pop(k, None)
        self._last_cleanup = now

    def _evict_excess_locked(self) -> None:
        if len(self._windows) < self._max_keys:
            return
        # Evict oldest-seen keys first to keep cache bounded
        overflow = len(self._windows) - self._max_keys + 1
        victim_keys = sorted(
            self._windows.keys(),
            key=lambda k: self._windows[k][-1] if self._windows[k] else 0.0,
        )[:overflow]
        for k in victim_keys:
            self._windows.pop(k, None)

    def check(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            if (
                now - self._last_cleanup >= self._cleanup_interval_seconds
                or len(self._windows) >= self._max_keys
            ):
                self._cleanup_locked(now, self._max_idle_seconds)

            if key not in self._windows:
                self._evict_excess_locked()

            timestamps = self._windows.setdefault(key, [])
            # Filter out timestamps outside the active window
            valid_timestamps = [t for t in timestamps if t > window_start]
            self._windows[key] = valid_timestamps

            count = len(valid_timestamps)
            if count >= limit:
                # Oldest timestamp determines reset time
                oldest = valid_timestamps[0] if valid_timestamps else now
                reset_after = max(1, int(oldest + window_seconds - now))
                return RateLimitResult(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_after=reset_after,
                )

            # Record current request
            self._windows[key].append(now)
            remaining = limit - count - 1
            reset_after = window_seconds
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_after=reset_after,
            )

    def cleanup(self, max_idle_seconds: int = 3600) -> None:
        """Explicit cleanup of stale keys older than `max_idle_seconds`."""
        now = time.time()
        with self._lock:
            self._cleanup_locked(now, max_idle_seconds)


class RedisRateLimiter:
    """
    Distributed sliding-window rate limiter backed by Redis sorted sets.
    Automatically falls back to `MemoryRateLimiter` when Redis is unreachable.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Any] = None,
        key_prefix: str = "ares:rl:",
        fallback_limiter: Optional[MemoryRateLimiter] = None,
        socket_timeout: float = 1.0,
    ):
        self._redis_url = (redis_url or "").strip() or None
        self._redis_client = redis_client
        self._key_prefix = key_prefix
        self._fallback = fallback_limiter or MemoryRateLimiter()
        self._socket_timeout = float(socket_timeout)
        self._degraded = False
        self._init_lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        if self._degraded or (self._redis_client is None and not self._redis_url):
            return "memory_fallback"
        return "redis"

    def _get_client(self) -> Optional[Any]:
        if self._redis_client is not None:
            return self._redis_client
        if not self._redis_url:
            return None
        with self._init_lock:
            if self._redis_client is not None:
                return self._redis_client
            try:
                import redis

                self._redis_client = redis.Redis.from_url(
                    self._redis_url,
                    socket_timeout=self._socket_timeout,
                    socket_connect_timeout=self._socket_timeout,
                    decode_responses=True,
                )
            except Exception as exc:
                logger.warning("Failed creating Redis rate limiter client (%s); using memory fallback", exc)
                self._degraded = True
                return None
        return self._redis_client

    def check(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        client = self._get_client()
        if client is None:
            self._degraded = True
            return self._fallback.check(key=key, limit=limit, window_seconds=window_seconds)

        now = time.time()
        window_start = now - window_seconds
        redis_key = f"{self._key_prefix}{key}"
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        ttl_ms = max(1000, int(window_seconds * 1000) + 1000)

        try:
            pipe = client.pipeline()
            pipe.zremrangebyscore(redis_key, 0, window_start)
            pipe.zcard(redis_key)
            pipe.zrange(redis_key, 0, 0, withscores=True)
            _, current_count, oldest_entries = pipe.execute()

            count = int(current_count or 0)
            if count >= limit:
                oldest_ts = now
                if oldest_entries:
                    first_entry = oldest_entries[0]
                    if isinstance(first_entry, (tuple, list)) and len(first_entry) >= 2:
                        oldest_ts = float(first_entry[1])
                reset_after = max(1, int(oldest_ts + window_seconds - now))
                self._degraded = False
                return RateLimitResult(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_after=reset_after,
                )

            write_pipe = client.pipeline()
            write_pipe.zadd(redis_key, {member: now})
            write_pipe.pexpire(redis_key, ttl_ms)
            write_pipe.execute()

            self._degraded = False
            remaining = max(0, limit - count - 1)
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_after=window_seconds,
            )
        except Exception as exc:
            logger.warning("Redis rate limiter unavailable (%s); falling back to MemoryRateLimiter", exc)
            self._degraded = True
            return self._fallback.check(key=key, limit=limit, window_seconds=window_seconds)


def create_rate_limiter(
    redis_url: Optional[str] = None,
) -> Union[MemoryRateLimiter, RedisRateLimiter]:
    """Instantiate RedisRateLimiter when `redis_url` is configured, else MemoryRateLimiter."""
    if redis_url and str(redis_url).strip():
        return RedisRateLimiter(redis_url=str(redis_url).strip())
    return MemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    HTTP Rate Limiting Middleware for FastAPI.
    Injects standard rate-limit headers (X-RateLimit-*) on all evaluated responses.
    """

    def __init__(
        self,
        app,
        limiter: Union[MemoryRateLimiter, RedisRateLimiter],
        enabled: bool = True,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.limiter = limiter
        self.enabled = enabled
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths or {
            "/healthz",
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    def _get_client_key(self, request: Request) -> str:
        """Derive rate-limit key by API key, then client IP."""
        # 1. API key if provided
        api_key = request.headers.get("X-Ares-Api-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:].strip()

        if api_key:
            return f"key:{api_key[:16]}"

        # 2. Client IP fallback
        client_host = request.client.host if request.client else "127.0.0.1"
        return f"ip:{client_host}"

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Bypass probe and doc endpoints
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        key = self._get_client_key(request)
        result = self.limiter.check(
            key=key,
            limit=self.requests_per_window,
            window_seconds=self.window_seconds,
        )

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded for %s on %s %s",
                key,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Too many requests. Limit is {result.limit} per {self.window_seconds}s.",
                    "retry_after_seconds": result.reset_after,
                    "retryable": True,
                },
            )
            response.headers["Retry-After"] = str(result.reset_after)
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(result.reset_after)
            return response

        # Proceed to route handler
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_after)
        return response
