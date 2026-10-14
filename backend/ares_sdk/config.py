"""
ARES SDK Configuration
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class AresConfig:
    api_key: str
    base_url: str = "http://localhost:8000"
    timeout: float = 30.0
    max_retries: int = 3
