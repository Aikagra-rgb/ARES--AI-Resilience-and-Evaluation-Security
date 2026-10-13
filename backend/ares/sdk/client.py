"""
ARES Python SDK Client.

Provides an asynchronous HTTP client with built-in HMAC-SHA256 request signing,
tenant context propagation, and structured response handling.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class AresClientError(Exception):
    """Exception raised for ARES client errors."""
    pass


class AresClient:
    """
    Client for interacting with the ARES Gateway and API.
    Supports HMAC signing, tenant routing, and error resilience.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        key_id: Optional[str] = None,
        secret: Optional[str] = None,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        timeout: float = 5.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.key_id = key_id
        self.secret = secret
        self.organization_id = organization_id
        self.project_id = project_id
        self.timeout = timeout
        self._external_client = http_client

    def _sign_request(
        self,
        method: str,
        path: str,
        body_bytes: bytes,
    ) -> Dict[str, str]:
        """Compute canonical HMAC-SHA256 headers matching ARES HmacVerifier."""
        if not self.key_id or not self.secret:
            return {}

        now_ts = str(int(time.time()))
        nonce = str(uuid.uuid4())
        body_sha256 = (
            hashlib.sha256(body_bytes).hexdigest().lower()
            if body_bytes
            else ""
        )

        canonical = f"{method.upper()}\n{path}\n\n{body_sha256}\n{now_ts}\n{nonce}"
        signature = hmac.new(
            self.secret.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest().lower()

        return {
            "X-Ares-Key-Id": self.key_id,
            "X-Ares-Timestamp": now_ts,
            "X-Ares-Nonce": nonce,
            "X-Ares-Signature": signature,
        }

    async def enforce(
        self,
        prompt: str,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
        canary_token: Optional[str] = None,
        block_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send an enforcement request to the ARES Gateway (/gateway/enforce).
        """
        path = "/gateway/enforce"
        url = f"{self.base_url}{path}"

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "tenant_id": tenant_id,
        }
        if session_id:
            payload["session_id"] = session_id
        if metadata:
            payload["metadata"] = metadata

        # Serialize payload
        import json
        body_bytes = json.dumps(payload).encode("utf-8")

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-Ares-Api-Key"] = self.api_key
        if self.organization_id:
            headers["X-Ares-Organization-Id"] = self.organization_id
        if self.project_id:
            headers["X-Ares-Project-Id"] = self.project_id
        if canary_token:
            headers["X-Ares-Canary-Token"] = canary_token
        if block_threshold is not None:
            headers["X-Ares-Block-Threshold"] = str(block_threshold)

        # Attach HMAC signature if credentials configured
        headers.update(self._sign_request("POST", path, body_bytes))

        try:
            if self._external_client:
                response = await self._external_client.post(
                    url,
                    content=body_bytes,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        content=body_bytes,
                        headers=headers,
                    )

            if response.status_code >= 400:
                raise AresClientError(
                    f"ARES Gateway returned HTTP {response.status_code}: {response.text}"
                )

            return response.json()
        except httpx.RequestError as exc:
            raise AresClientError(f"Failed to communicate with ARES Gateway: {exc}") from exc
