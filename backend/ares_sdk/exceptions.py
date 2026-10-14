"""
ARES SDK Exceptions
"""

class AresError(Exception):
    """Base exception for all ARES SDK errors."""
    pass

class AresAPIError(AresError):
    """Raised when an API call returns a non-200 error."""
    pass

class AresAuthenticationError(AresError):
    """Raised when authentication fails."""
    pass

class AresRateLimitError(AresError):
    """Raised when rate limits are exceeded."""
    pass
