"""
ARES Python SDK.

Provides client libraries and ASGI middleware for integrating ARES
security gateway inspection into modern LLM applications.
"""

from ares.sdk.client import AresClient, AresClientError
from ares.sdk.middleware import (
    AresSecurityMiddleware,
    default_prompt_extractor,
    default_prompt_sanitizer,
)

__all__ = [
    "AresClient",
    "AresClientError",
    "AresSecurityMiddleware",
    "default_prompt_extractor",
    "default_prompt_sanitizer",
]
