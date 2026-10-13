"""
ARES Server-Side HMAC-SHA256 Request Verification Engine.

Provides cryptographic request integrity, authenticity, and replay-attack
prevention compatible with the C# FastApiAresApiClient (.NET).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Dict, Optional, Set, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class HmacVerifier:
    """
    Validates HMAC-SHA256 signatures and guards against replay attacks.

    Canonical Request format (newline-separated):
    ---------------------------------------------
    1. HTTP Method in UPPERCASE (e.g. 'POST')
    2. Request path (e.g. '/api/gateway/enforce')
    3. Query string without leading '?' (e.g. 'limit=10')
    4. SHA256 hex digest of request body (empty string if no body)
    5. Unix timestamp in seconds (string)
    6. Unique nonce string
    """

    def __init__(
        self,
        key_store: Optional[Dict[str, str]] = None,
        default_secret: Optional[str] = None,
        max_clock_skew_seconds: int = 300,
    ):
        self.key_store: Dict[str, str] = dict(key_store) if key_store else {}
        self.default_secret: Optional[str] = default_secret
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self._nonce_cache: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_key(self, key_id: str, secret: str) -> None:
        """Register or update an HMAC signing key."""
        with self._lock:
            self.key_store[key_id] = secret

    def _purge_expired_nonces(self, now: float) -> None:
        """Purge nonces whose validity window has elapsed."""
        expired = [nonce for nonce, exp in self._nonce_cache.items() if exp < now]
        for nonce in expired:
            self._nonce_cache.pop(nonce, None)

    def verify(
        self,
        method: str,
        path: str,
        query: str,
        body: bytes | str,
        timestamp_str: str,
        nonce: str,
        signature: str,
        key_id: str = "ares-web",
    ) -> Tuple[bool, str]:
        """
        Verify request signature, clock drift, and nonce freshness.

        Returns:
            (is_valid, reason_or_message)
        """
        if not signature:
            return False, "Missing signature"
        if not timestamp_str:
            return False, "Missing timestamp"
        if not nonce:
            return False, "Missing nonce"

        # 1. Resolve secret key
        secret = self.key_store.get(key_id) or self.default_secret
        if not secret:
            return False, f"Unknown key_id '{key_id}' and no default HMAC secret configured"

        # 2. Clock skew validation
        try:
            req_time = int(timestamp_str)
        except (ValueError, TypeError):
            return False, "Malformed timestamp format (must be unix epoch integer)"

        now = int(time.time())
        if abs(now - req_time) > self.max_clock_skew_seconds:
            return (
                False,
                f"Clock skew exceeded: request timestamp {req_time} differs from server time {now} by >{self.max_clock_skew_seconds}s",
            )

        # 3. Replay attack prevention (nonce tracking)
        with self._lock:
            self._purge_expired_nonces(now)
            if nonce in self._nonce_cache:
                return False, f"Replay attack detected: nonce '{nonce}' has already been processed"
            # Keep nonce alive for duration of the allowed clock skew window
            self._nonce_cache[nonce] = req_time + self.max_clock_skew_seconds

        # 4. Canonical payload construction
        body_bytes = body.encode("utf-8") if isinstance(body, str) else (body or b"")
        body_sha256 = (
            hashlib.sha256(body_bytes).hexdigest().lower()
            if body_bytes
            else ""
        )
        canonical_parts = [
            method.upper(),
            path,
            query.lstrip("?"),
            body_sha256,
            str(timestamp_str),
            str(nonce),
        ]
        canonical = "\n".join(canonical_parts)

        # 5. Compute HMAC-SHA256
        computed = hmac.new(
            secret.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().lower()

        # 6. Constant-time comparison
        if not hmac.compare_digest(computed, signature.lower()):
            return False, "HMAC signature mismatch"

        return True, "Valid"


class HmacVerificationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / Starlette middleware enforcing HMAC request signatures.
    """

    def __init__(
        self,
        app,
        verifier: HmacVerifier,
        enforce_mode: str = "permissive",  # 'enforcing', 'permissive', 'disabled'
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.verifier = verifier
        self.enforce_mode = enforce_mode
        self.exempt_paths = exempt_paths or {
            "/healthz",
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        if self.enforce_mode == "disabled":
            return await call_next(request)

        # Bypass health / doc paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Extract headers
        signature = request.headers.get("X-Ares-Signature")
        timestamp = request.headers.get("X-Ares-Timestamp")
        nonce = request.headers.get("X-Ares-Nonce")
        key_id = request.headers.get("X-Ares-Key-Id", "ares-web")

        has_sig_headers = bool(signature and timestamp and nonce)

        # In permissive mode, only verify if headers are present
        if not has_sig_headers:
            if self.enforce_mode == "enforcing":
                return JSONResponse(
                    status_code=401,
                    content={
                        "code": "HMAC_AUTH_REQUIRED",
                        "message": "Missing HMAC signature headers (X-Ares-Signature, X-Ares-Timestamp, X-Ares-Nonce)",
                        "retryable": False,
                    },
                )
            return await call_next(request)

        # Read body and cache for downstream use
        body_bytes = await request.body()

        # Reset receive callable so downstream endpoints can read request.body()
        async def receive():
            return {"type": "http.request", "body": body_bytes}

        request._receive = receive

        # Perform verification
        is_valid, reason = self.verifier.verify(
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            body=body_bytes,
            timestamp_str=timestamp,
            nonce=nonce,
            signature=signature,
            key_id=key_id,
        )

        if not is_valid:
            logger.warning(
                "HMAC verification failed for %s %s [%s]: %s",
                request.method,
                request.url.path,
                key_id,
                reason,
            )
            return JSONResponse(
                status_code=403 if "mismatch" in reason else 401,
                content={
                    "code": "HMAC_SIGNATURE_INVALID",
                    "message": f"HMAC signature verification failed: {reason}",
                    "retryable": False,
                },
            )

        return await call_next(request)
