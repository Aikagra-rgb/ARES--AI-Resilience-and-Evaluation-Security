"""
ARES HTTP Gateway Rate Limiting Engine.

Provides token bucket and sliding window rate limiting at the FastAPI application layer.
Supports both a thread-safe in-memory sliding window backend and a distributed Redis backend.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

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
    Thread-safe in-memory sliding-window log rate limiter.
    """

    def __init__(self):
        self._windows: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> RateLimitResult:
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
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
        """Periodic cleanup of stale keys."""
        now = time.time()
        cutoff = now - max_idle_seconds
        with self._lock:
            stale_keys = [
                k for k, timestamps in self._windows.items()
                if not timestamps or timestamps[-1] < cutoff
            ]
            for k in stale_keys:
                self._windows.pop(k, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    HTTP Rate Limiting Middleware for FastAPI.
    Injects standard rate-limit headers (X-RateLimit-*) on all evaluated responses.
    """

    def __init__(
        self,
        app,
        limiter: MemoryRateLimiter,
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
