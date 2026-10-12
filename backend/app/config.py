"""
Legacy compatibility module for app.config.

This module provides backward compatibility by re-exporting the unified
configuration from ares.config.
"""

from functools import lru_cache
from ares.config import Settings, get_settings

# Re-export for backward compatibility
__all__ = ["Settings", "get_settings", "get_app_settings"]


# Legacy alias
get_app_settings = get_settings


# Re-export the Settings class from ares
# This ensures any code using `from app.config import Settings` gets the unified class
Settings = Settings