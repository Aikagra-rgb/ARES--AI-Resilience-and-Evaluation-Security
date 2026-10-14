"""
ARES Dynamic HMAC Key Management & Lifecycle Engine
Supports automated cryptographic key generation, rotation, expiration, and validation.
"""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

class HMACKeyManager:
    """
    Manages dynamic HMAC API credentials, key lifecycle rotation, and signature verification.
    """

    def __init__(self):
        self.keys: Dict[str, dict] = {}

    def generate_key(self, key_id: str, rotate_after_days: int = 30) -> str:
        """Generate and register a new cryptographically secure HMAC secret."""
        secret = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        self.keys[key_id] = {
            "secret": secret,
            "created_at": now,
            "expires_at": now + timedelta(days=rotate_after_days),
            "status": "active",
        }
        return secret

    def rotate_key(self, key_id: str, rotate_after_days: int = 30) -> str:
        """Rotate existing key by retiring old secret and provisioning new one."""
        if key_id in self.keys:
            self.keys[key_id]["status"] = "retired"
        return self.generate_key(key_id, rotate_after_days)

    def revoke_key(self, key_id: str) -> bool:
        """Revoke key immediately."""
        if key_id in self.keys:
            self.keys[key_id]["status"] = "revoked"
            return True
        return False

    def validate_signature(
        self,
        key_id: str,
        signature: str,
        method: str,
        path: str,
        body: str,
        timestamp: str,
        nonce: str,
    ) -> bool:
        """Validate canonical HMAC-SHA256 signature against active key store."""
        key_record = self.keys.get(key_id)
        if not key_record or key_record["status"] != "active":
            return False

        if datetime.utcnow() > key_record["expires_at"]:
            key_record["status"] = "expired"
            return False

        canonical = f"{method.upper()}\n{path}\n{body}\n{timestamp}\n{nonce}"
        computed = hmac.new(
            key_record["secret"].encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed.lower(), signature.lower())
