"""
Shared Error Message & Log Scrubber for Credentials and PII.

Used across runtime gateway error responses and circuit-breaker warnings to
prevent upstream API keys, Bearer tokens, private keys, database URIs, and PII
from leaking into logs or caller responses.
"""

from __future__ import annotations

import re
from typing import List, Tuple

OUTBOUND_REDACTION_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
        "private_key_leak",
    ),
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9_]{30,}|gsk_[A-Za-z0-9_-]{24,}|nvapi-[A-Za-z0-9_-]{20,}|ares_tnt_[A-Za-z0-9_-]{8,})\b"
        ),
        "[REDACTED_API_KEY]",
        "api_key_leak",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[REDACTED_JWT_TOKEN]",
        "jwt_token_leak",
    ),
    (
        re.compile(r"(?:postgres|postgresql|mysql|mongodb|redis)://[^\s\"'<>]+"),
        "[REDACTED_DB_URI]",
        "database_uri_leak",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
        "ssn_pii_leak",
    ),
    (
        re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[ -]?\d{4}[ -]?\d{4}[ -]?\d{3,4}\b"),
        "[REDACTED_CREDIT_CARD]",
        "credit_card_pii_leak",
    ),
]

BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")


def sanitize_error_message(
    message: object,
    max_len: int = 500,
    default_message: str = "Upstream LLM provider error.",
) -> str:
    """Redact credentials/PII and bound length on error strings before logging or returning."""
    cleaned = str(message or "").strip()
    if not cleaned:
        return default_message
    for pattern, replacement, _ in OUTBOUND_REDACTION_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED_TOKEN]", cleaned)
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned
