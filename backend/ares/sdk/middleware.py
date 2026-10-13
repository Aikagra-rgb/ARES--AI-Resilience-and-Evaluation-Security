"""
ARES SDK Client Middleware for ASGI and FastAPI applications.

Provides seamless drop-in security inspection, automated prompt extraction,
fail-closed/fail-open resilience, request sanitization, and violation blocking.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ares.api.schemas import EnforceActionEnum, EnforceRequest
from ares.gateway.context import TenantContext
from ares.gateway.enforcer import RuntimeEnforcer
from ares.sdk.client import AresClient, AresClientError

logger = logging.getLogger(__name__)


def default_prompt_extractor(body_json: Dict[str, Any]) -> Optional[str]:
    """
    Extract prompt text from standard LLM payloads:
    1. OpenAI Chat: messages -> last user message
    2. Prompt / Text / Input / Query top-level fields
    """
    if not isinstance(body_json, dict):
        return None

    # OpenAI Chat format
    if "messages" in body_json and isinstance(body_json["messages"], list):
        for msg in reversed(body_json["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Multi-modal / chunked text
                    texts = [
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    if texts:
                        return " ".join(texts)

    # Standard completion / generation formats
    for field in ("prompt", "input", "text", "query", "message", "content"):
        val = body_json.get(field)
        if isinstance(val, str) and val.strip():
            return val

    return None


def default_prompt_sanitizer(
    body_json: Dict[str, Any], sanitized_text: str
) -> Dict[str, Any]:
    """Replace original prompt with sanitized text in body dictionary."""
    updated = dict(body_json)
    if "messages" in updated and isinstance(updated["messages"], list):
        messages = list(updated["messages"])
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg_copy = dict(msg)
                msg_copy["content"] = sanitized_text
                messages[idx] = msg_copy
                break
        updated["messages"] = messages
        return updated

    for field in ("prompt", "input", "text", "query", "message", "content"):
        if field in updated and isinstance(updated[field], str):
            updated[field] = sanitized_text
            return updated

    return updated


class AresSecurityMiddleware:
    """
    ASGI middleware intercepting incoming requests, extracting LLM prompt payloads,
    and delegating enforcement to ARES (either via remote HTTP client or in-process enforcer).
    """

    def __init__(
        self,
        app: Any,
        client: Optional[AresClient] = None,
        enforcer: Optional[RuntimeEnforcer] = None,
        protected_paths: Optional[List[str]] = None,
        excluded_paths: Optional[List[str]] = None,
        fail_closed: bool = True,
        tenant_id: str = "default",
        prompt_extractor: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
        prompt_sanitizer: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    ):
        self.app = app
        self.client = client
        self.enforcer = enforcer
        self.protected_paths = set(protected_paths) if protected_paths else None
        self.excluded_paths = set(
            excluded_paths
            or ["/healthz", "/metrics", "/docs", "/openapi.json", "/favicon.ico"]
        )
        self.fail_closed = fail_closed
        self.tenant_id = tenant_id
        self.prompt_extractor = prompt_extractor or default_prompt_extractor
        self.prompt_sanitizer = prompt_sanitizer or default_prompt_sanitizer

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")

        # Only inspect mutation requests that carry request bodies
        if method not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return

        if path in self.excluded_paths:
            await self.app(scope, receive, send)
            return

        if self.protected_paths is not None and path not in self.protected_paths:
            await self.app(scope, receive, send)
            return

        # Buffer incoming request body
        body_chunks: List[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        raw_body = b"".join(body_chunks)

        # Attempt to parse body as JSON
        body_json: Optional[Dict[str, Any]] = None
        prompt_text: Optional[str] = None
        if raw_body:
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
                if isinstance(parsed, dict):
                    body_json = parsed
                    prompt_text = self.prompt_extractor(body_json)
            except Exception:
                pass

        # If no inspectable prompt found, pass through unmodified
        if not prompt_text:
            replay_receive = self._create_replay_receive(raw_body)
            await self.app(scope, replay_receive, send)
            return

        # Execute ARES inspection
        try:
            action, risk_score, severity, reasons, sanitized_text = await self._inspect(
                prompt_text=prompt_text,
                scope=scope,
            )
        except Exception as exc:
            logger.error("ARES inspection failed in SDK middleware: %s", exc)
            if self.fail_closed:
                await self._send_json_response(
                    send,
                    status=503,
                    payload={
                        "detail": "ARES Security inspection unavailable (fail-closed mode active)",
                        "error": str(exc),
                    },
                    extra_headers=[(b"x-ares-status", b"fail-closed-error")],
                )
                return
            else:
                # Fail-open: forward request with degraded telemetry header
                replay_receive = self._create_replay_receive(raw_body)
                wrapped_send = self._create_wrapped_send(
                    send,
                    headers_to_add=[(b"x-ares-degraded", b"true")],
                )
                await self.app(scope, replay_receive, wrapped_send)
                return

        # Enforce action decisions
        if action == "block":
            await self._send_json_response(
                send,
                status=403,
                payload={
                    "detail": "Request blocked by ARES Security Gateway",
                    "action": "block",
                    "severity": severity,
                    "risk_score": risk_score,
                    "reasons": reasons,
                },
                extra_headers=[
                    (b"x-ares-action", b"block"),
                    (b"x-ares-risk-score", str(risk_score).encode("utf-8")),
                ],
            )
            return

        if action == "quarantine":
            await self._send_json_response(
                send,
                status=423,
                payload={
                    "detail": "Request quarantined by ARES Security Gateway for manual review",
                    "action": "quarantine",
                    "severity": severity,
                    "risk_score": risk_score,
                    "reasons": reasons,
                },
                extra_headers=[
                    (b"x-ares-action", b"quarantine"),
                    (b"x-ares-risk-score", str(risk_score).encode("utf-8")),
                ],
            )
            return

        # If action is sanitize, rewrite the body
        active_body = raw_body
        if action == "sanitize" and sanitized_text and body_json is not None:
            sanitized_dict = self.prompt_sanitizer(body_json, sanitized_text)
            active_body = json.dumps(sanitized_dict).encode("utf-8")
            # Update Content-Length header in scope
            new_headers = []
            for h_name, h_val in scope.get("headers", []):
                if h_name.lower() == b"content-length":
                    new_headers.append((b"content-length", str(len(active_body)).encode("utf-8")))
                else:
                    new_headers.append((h_name, h_val))
            scope["headers"] = new_headers

        replay_receive = self._create_replay_receive(active_body)
        wrapped_send = self._create_wrapped_send(
            send,
            headers_to_add=[
                (b"x-ares-action", action.encode("utf-8")),
                (b"x-ares-risk-score", str(risk_score).encode("utf-8")),
            ],
        )
        await self.app(scope, replay_receive, wrapped_send)

    async def _inspect(
        self,
        prompt_text: str,
        scope: Dict[str, Any],
    ) -> Tuple[str, float, str, List[str], Optional[str]]:
        """
        Execute inspection via in-process RuntimeEnforcer or remote AresClient.
        Returns: (action, risk_score, severity, reasons, sanitized_text)
        """
        if self.enforcer:
            # Co-located in-process enforcement
            req = EnforceRequest(prompt=prompt_text, tenant_id=self.tenant_id)
            tenant_ctx = TenantContext(tenant_id=self.tenant_id)
            decision = await self.enforcer.enforce(req, context=tenant_ctx)
            action_str = (
                decision.action.value
                if isinstance(decision.action, EnforceActionEnum)
                else str(decision.action)
            )
            return (
                action_str,
                float(decision.risk_score),
                decision.severity,
                decision.reasons or [],
                decision.sanitized_text,
            )

        if self.client:
            # Remote HTTP gateway enforcement
            result = await self.client.enforce(prompt=prompt_text, tenant_id=self.tenant_id)
            return (
                result.get("action", "allow"),
                float(result.get("risk_score", 0.0)),
                result.get("severity", "safe"),
                result.get("reasons", []),
                result.get("sanitized_text"),
            )

        # Default fallback if neither client nor enforcer provided
        return ("allow", 0.0, "safe", [], None)

    def _create_replay_receive(self, body_bytes: bytes) -> Callable:
        """Create an ASGI receive callable that replays buffered body bytes."""
        has_sent = False

        async def replay_receive() -> Dict[str, Any]:
            nonlocal has_sent
            if not has_sent:
                has_sent = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }

        return replay_receive

    def _create_wrapped_send(
        self, send: Callable, headers_to_add: List[Tuple[bytes, bytes]]
    ) -> Callable:
        """Wrap ASGI send to inject ARES security telemetry headers into http.response.start."""
        async def wrapped_send(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(headers_to_add)
                message = dict(message)
                message["headers"] = headers
            await send(message)

        return wrapped_send

    async def _send_json_response(
        self,
        send: Callable,
        status: int,
        payload: Dict[str, Any],
        extra_headers: Optional[List[Tuple[bytes, bytes]]] = None,
    ) -> None:
        """Send a structured JSON HTTP response directly over ASGI."""
        body = json.dumps(payload).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("utf-8")),
        ]
        if extra_headers:
            headers.extend(extra_headers)

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )
