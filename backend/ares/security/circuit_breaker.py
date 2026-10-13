"""
ARES Circuit Breaker Engine for External Dependencies.

Protects the runtime gateway against cascading failures and thread exhaustion
when external upstream dependencies (Qdrant Cloud, Gemini Embeddings, Groq LLM)
experience network degradation or outages.
"""

from __future__ import annotations

import asyncio
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through
    OPEN = "OPEN"            # Tripped: fast-fails immediately without external call
    HALF_OPEN = "HALF_OPEN"  # Probe mode: allows single trial request to test recovery


class CircuitBreakerOpenError(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    def __init__(self, service_name: str, reset_after: float):
        super().__init__(
            f"Circuit breaker for '{service_name}' is OPEN. "
            f"Fast-failing to protect gateway. Retry in {reset_after:.1f}s."
        )
        self.service_name = service_name
        self.reset_after = reset_after


class CircuitBreaker:
    """
    Thread-safe & async-compatible Circuit Breaker.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.time()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            now = time.time()
            if self._state == CircuitState.OPEN:
                if now - self._last_state_change >= self.recovery_timeout_seconds:
                    logger.info(
                        "Circuit breaker '%s' transition: OPEN -> HALF_OPEN (probing recovery)",
                        self.name,
                    )
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = now
            return self._state

    def record_success(self) -> None:
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(
                    "Circuit breaker '%s' recovered: %s -> CLOSED",
                    self.name,
                    self._state.value,
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_state_change = time.time()

    def record_failure(self, error: Exception) -> None:
        with self._lock:
            now = time.time()
            self._failure_count += 1
            logger.warning(
                "Circuit breaker '%s' recorded failure (%d/%d): %s",
                self.name,
                self._failure_count,
                self.failure_threshold,
                error,
            )

            if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.error(
                        "Circuit breaker '%s' TRIPPED to OPEN state. All calls will fast-fail for %.1fs.",
                        self.name,
                        self.recovery_timeout_seconds,
                    )
                self._state = CircuitState.OPEN
                self._last_state_change = now

    async def execute(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute an async operation with circuit-breaker protection.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            time_remaining = max(
                0.0,
                self.recovery_timeout_seconds - (time.time() - self._last_state_change),
            )
            raise CircuitBreakerOpenError(self.name, time_remaining)

        try:
            result = await operation(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure(exc)
            raise


# Global registry of circuit breakers
_BREAKERS: Dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 3,
    recovery_timeout_seconds: float = 30.0,
) -> CircuitBreaker:
    with _REGISTRY_LOCK:
        if name not in _BREAKERS:
            _BREAKERS[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )
        return _BREAKERS[name]
