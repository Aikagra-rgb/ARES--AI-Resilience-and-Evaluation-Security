"""
ASGI Security Middleware for FastAPI / Starlette
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing import Callable, Optional

class AresSecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces ARES API Key verification and fail-closed ingress security.
    """

    def __init__(self, app, api_key: str, fail_closed: bool = True):
        super().__init__(app)
        self.api_key = api_key
        self.fail_closed = fail_closed

    async def dispatch(self, request: Request, call_next: Callable):
        # Allow unrestricted health checks
        if request.url.path in ("/health", "/healthz", "/docs", "/openapi.json"):
            return await call_next(request)

        provided_key = request.headers.get("X-Ares-Api-Key")
        if not provided_key or provided_key != self.api_key:
            status_code = 403 if self.fail_closed else 401
            return JSONResponse(
                status_code=status_code,
                content={"error": "Invalid or missing X-Ares-Api-Key header", "status": "blocked"},
            )

        return await call_next(request)
