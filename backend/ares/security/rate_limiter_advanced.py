"""
Advanced Sliding Window Rate Limiter
Supports Redis clustering with automatic in-memory sliding window fallback.
"""
from collections import defaultdict
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedRateLimiter:
    """Sliding-window rate limiter utilizing Redis with local memory fallback."""

    def __init__(self, redis_url: Optional[str] = "redis://localhost:6379"):
        self.redis = None
        self._memory_store: Dict[str, List[float]] = defaultdict(list)
        if redis_url:
            try:
                import redis
                r = redis.from_url(redis_url, socket_timeout=0.5)
                r.ping()
                self.redis = r
                logger.info("Connected to Redis for distributed rate limiting")
            except Exception:
                logger.info("Redis not available; using in-memory sliding window fallback")

    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        """
        Returns True if request is allowed, False if rate limit is exceeded.
        """
        now = time.time()
        window_start = now - window_seconds

        if self.redis is not None:
            try:
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window_seconds + 5)
                results = pipe.execute()
                count = results[2]
                return count <= limit
            except Exception:
                pass  # Fall through to in-memory

        # In-memory sliding window
        timestamps = self._memory_store[key]
        self._memory_store[key] = [t for t in timestamps if t > window_start]
        self._memory_store[key].append(now)
        return len(self._memory_store[key]) <= limit
