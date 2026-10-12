"""
ARES Request Signing Module.

Provides HMAC-based request signing for API authentication and integrity verification.
Supports both client-side signing and server-side verification.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlencode

from ares.config import Settings, settings as default_settings


@dataclass
class SignedRequest:
    """A request with HMAC signature."""
    method: str
    path: str
    query_params: Dict[str, str]
    body: Optional[str]
    timestamp: int
    nonce: str
    signature: str
    key_id: str
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for transmission."""
        return {
            "X-Ares-Timestamp": str(self.timestamp),
            "X-Ares-Nonce": self.nonce,
            "X-Ares-Signature": self.signature,
            "X-Ares-Key-Id": self.key_id,
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str], method: str, path: str, 
                     query_params: Dict[str, str], body: Optional[str]) -> "SignedRequest":
        """Reconstruct SignedRequest from HTTP headers."""
        return cls(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
            timestamp=int(headers.get("X-Ares-Timestamp", "0")),
            nonce=headers.get("X-Ares-Nonce", ""),
            signature=headers.get("X-Ares-Signature", ""),
            key_id=headers.get("X-Ares-Key-Id", ""),
        )


class RequestSigner:
    """
    HMAC-based request signer for ARES API.
    
    Uses HMAC-SHA256 with a shared secret key. The signature covers:
    - HTTP method
    - Request path
    - Query parameters (sorted)
    - Request body (if present)
    - Timestamp (to prevent replay attacks)
    - Nonce (to prevent replay attacks)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        signing_key: Optional[str] = None,
        key_id: str = "default",
        max_timestamp_drift: int = 300,  # 5 minutes
    ):
        self.settings = settings or default_settings
        self._signing_key = signing_key or self._derive_signing_key()
        self._key_id = key_id
        self._max_timestamp_drift = max_timestamp_drift
    
    def _derive_signing_key(self) -> str:
        """Derive signing key from existing secrets."""
        # Use a combination of existing secrets to derive a signing key
        import hashlib
        source = (
            (self.settings.api_key or "") +
            (self.settings.prompt_fingerprint_secret or "") +
            (self.settings.execution_payload_encryption_key.get_secret_value() if self.settings.execution_payload_encryption_key else "")
        )
        return hashlib.sha256(source.encode()).hexdigest()
    
    def _canonical_request(
        self,
        method: str,
        path: str,
        query_params: Dict[str, str],
        body: Optional[str],
        timestamp: int,
        nonce: str,
    ) -> str:
        """Create canonical request string for signing."""
        # Sort query params for consistent ordering
        sorted_query = urlencode(sorted(query_params.items())) if query_params else ""
        
        # Hash body if present
        body_hash = ""
        if body:
            body_hash = hashlib.sha256(body.encode()).hexdigest()
        
        parts = [
            method.upper(),
            path,
            sorted_query,
            body_hash,
            str(timestamp),
            nonce,
        ]
        return "\n".join(parts)
    
    def sign_request(
        self,
        method: str,
        path: str,
        query_params: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timestamp: Optional[int] = None,
        nonce: Optional[str] = None,
    ) -> SignedRequest:
        """
        Sign a request and return SignedRequest object.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g., /api/v1/harden)
            query_params: Query parameters
            body: Request body as string
            timestamp: Unix timestamp (defaults to current time)
            nonce: Unique nonce (defaults to random)
            
        Returns:
            SignedRequest with signature and headers
        """
        ts = timestamp or int(time.time())
        nc = nonce or self._generate_nonce()
        query = query_params or {}
        
        canonical = self._canonical_request(method, path, query, body, ts, nc)
        
        signature = hmac.new(
            self._signing_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return SignedRequest(
            method=method,
            path=path,
            query_params=query,
            body=body,
            timestamp=ts,
            nonce=nc,
            signature=signature,
            key_id=self._key_id,
        )
    
    def verify_request(self, signed_request: SignedRequest) -> tuple[bool, str]:
        """
        Verify a signed request.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check timestamp drift
        now = int(time.time())
        if abs(now - signed_request.timestamp) > self._max_timestamp_drift:
            return False, f"Timestamp drift too large: {abs(now - signed_request.timestamp)}s > {self._max_timestamp_drift}s"
        
        # Recreate canonical request and verify signature
        canonical = self._canonical_request(
            signed_request.method,
            signed_request.path,
            signed_request.query_params,
            signed_request.body,
            signed_request.timestamp,
            signed_request.nonce,
        )
        
        expected_signature = hmac.new(
            self._signing_key.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_signature, signed_request.signature):
            return False, "Invalid signature"
        
        return True, ""
    
    def _generate_nonce(self) -> str:
        """Generate a cryptographically secure nonce."""
        import secrets
        return secrets.token_hex(16)


# Global signer instance
_signer: Optional[RequestSigner] = None


def get_signer() -> RequestSigner:
    """Get or create the global request signer."""
    global _signer
    if _signer is None:
        _signer = RequestSigner()
    return _signer


def sign_request(
    method: str,
    path: str,
    query_params: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
) -> SignedRequest:
    """Convenience function to sign a request."""
    return get_signer().sign_request(method, path, query_params, body)


def verify_request(signed_request: SignedRequest) -> tuple[bool, str]:
    """Convenience function to verify a signed request."""
    return get_signer().verify_request(signed_request)