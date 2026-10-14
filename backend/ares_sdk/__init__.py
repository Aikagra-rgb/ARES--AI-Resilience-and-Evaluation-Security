"""
ARES Python SDK
Enterprise Resilience & Prompt Optimization Suite
"""

from ares_sdk.client import AresClient
from ares_sdk.config import AresConfig
from ares_sdk.exceptions import (
    AresAPIError,
    AresAuthenticationError,
    AresError,
    AresRateLimitError,
)
from ares_sdk.middleware import AresSecurityMiddleware

__version__ = "1.0.0"

__all__ = [
    "AresClient",
    "AresConfig",
    "AresError",
    "AresAPIError",
    "AresAuthenticationError",
    "AresRateLimitError",
    "AresSecurityMiddleware",
    "__version__",
]
