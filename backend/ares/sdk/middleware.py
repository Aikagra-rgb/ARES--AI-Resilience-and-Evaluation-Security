"""
ARES Security SDK - Embedded Application-Layer Defense.

Provides defense-in-depth by embedding security checks directly in protected applications.
This bypasses the gateway SPOF and enables sub-millisecond local decisions.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from collections import OrderedDict

from ares.security.structural_detector import (
    StructuralAttackDetector,
    DetectionResult,
    Decision,
    SeverityLevel,
    AttackType,
    Detection,
)

logger = logging.getLogger(__name__)


class CachePolicy(str, Enum):
    """Cache behavior for detection results."""
    ALLOW = "allow"      # Cache ALLOW decisions
    DENY = "deny"        # Cache BLOCK/CHALLENGE decisions
    ALL = "all"          # Cache all decisions
    NONE = "none"        # No caching


@dataclass
class SecurityConfig:
    """Configuration for embedded security middleware."""
    # Structural detector settings
    enabled_categories: Optional[List[AttackType]] = None
    severity_threshold: SeverityLevel = SeverityLevel.LOW
    
    # Cache settings
    cache_policy: CachePolicy = CachePolicy.DENY
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Gateway fallback
    gateway_enabled: bool = True
    gateway_url: str = "http://localhost:8000"
    gateway_timeout_ms: int = 5000
    gateway_auth_header: Optional[str] = None
    
    # Consensus settings
    require_consensus: bool = True
    local_weight: float = 0.6
    gateway_weight: float = 0.4
    
    # Async behavior
    async_gateway: bool = True  # Non-blocking gateway calls
    max_concurrent_gateway: int = 10
    
    # Logging
    log_decisions: bool = True
    log_level: int = logging.INFO


@dataclass
class ConversationContext:
    """Context for multi-turn conversation analysis."""
    conversation_id: str
    turn_number: int = 1
    previous_decisions: List[Decision] = field(default_factory=list)
    user_id: Optional[str] = None
    application_name: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityDecision:
    """Final security decision with full reasoning."""
    decision: Decision
    local_result: DetectionResult
    gateway_result: Optional[DetectionResult] = None
    consensus: bool = True
    latency_ms: float = 0.0
    cache_hit: bool = False
    correlation_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "local_result": self.local_result.to_dict(),
            "gateway_result": self.gateway_result.to_dict() if self.gateway_result else None,
            "consensus": self.consensus,
            "latency_ms": round(self.latency_ms, 2),
            "cache_hit": self.cache_hit,
            "correlation_id": self.correlation_id,
        }


class LRUCache:
    """Thread-safe LRU cache with TTL."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple] = OrderedDict()
    
    def _make_key(self, text: str, context: Optional[ConversationContext] = None) -> str:
        """Generate cache key from text and context."""
        content = text
        if context:
            content = f"{context.conversation_id}:{context.turn_number}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get(self, text: str, context: Optional[ConversationContext] = None) -> Optional[DetectionResult]:
        key = self._make_key(text, context)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return result
            else:
                # Expired
                del self._cache[key]
        return None
    
    def set(self, text: str, result: DetectionResult, context: Optional[ConversationContext] = None) -> None:
        key = self._make_key(text, context)
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove LRU
        self._cache[key] = (result, time.time())
    
    def clear(self) -> None:
        self._cache.clear()
    
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, "_hits", 0) / max(getattr(self, "_requests", 1), 1),
        }


class GatewayClient:
    """Async client for ARES Gateway API."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._session = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_gateway)
    
    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.gateway_timeout_ms / 1000)
            headers = {}
            if self.config.gateway_auth_header:
                headers["Authorization"] = self.config.gateway_auth_header
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session
    
    async def enforce(self, prompt: str, context: ConversationContext) -> Optional[DetectionResult]:
        """Call gateway for semantic analysis."""
        if not self.config.gateway_enabled:
            return None
        
        async with self._semaphore:
            try:
                session = await self._get_session()
                payload = {
                    "prompt": prompt,
                    "application_name": context.application_name,
                    "correlation_id": f"sdk-{context.conversation_id}-{context.turn_number}",
                }
                
                async with session.post(
                    f"{self.config.gateway_url}/api/gateway/enforce",
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._convert_gateway_response(data)
                    else:
                        logger.warning("Gateway returned %d: %s", resp.status, await resp.text())
                        return None
            except asyncio.TimeoutError:
                logger.warning("Gateway timeout for conversation %s", context.conversation_id)
                return None
            except Exception as e:
                logger.error("Gateway error: %s", e)
                return None
    
    def _convert_gateway_response(self, data: dict) -> DetectionResult:
        """Convert gateway response to DetectionResult."""
        # Map gateway action to our decision
        action_map = {
            "ALLOW": Decision.ALLOW,
            "SANITIZE": Decision.CHALLENGE,
            "BLOCK": Decision.BLOCK,
        }
        
        detections = []
        if data.get("matched_technique"):
            from ares.security.structural_detector import Detection
            detections.append(Detection(
                attack_type=AttackType.INSTRUCTION_OVERRIDE,  # Default
                pattern_name=data["matched_technique"],
                matched_text="",
                position=0,
                severity=SeverityLevel.HIGH if data["action"] == "BLOCK" else SeverityLevel.MEDIUM,
                confidence=min(data.get("risk_score", 50) / 100, 1.0),
                description=data.get("explanation", "Gateway detection"),
                rule_id="ARS-GATEWAY-001",
            ))
        
        result = DetectionResult()
        for d in detections:
            result.add_detection(d)
        result.decision = action_map.get(data.get("action", "ALLOW"), Decision.ALLOW)
        return result
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class ConsensusEngine:
    """Combines local and gateway decisions using weighted voting."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def decide(
        self,
        local: DetectionResult,
        gateway: Optional[DetectionResult],
    ) -> tuple[Decision, bool]:
        """
        Returns (final_decision, consensus_reached).
        
        Consensus = both agree or gateway unavailable.
        No consensus = escalate to human / BLOCK by default.
        """
        if gateway is None:
            # No gateway - trust local with slight penalty
            return local.decision, True
        
        if local.decision == gateway.decision:
            return local.decision, True
        
        # Weighted decision
        local_score = self._decision_score(local.decision) * self.config.local_weight
        gateway_score = self._decision_score(gateway.decision) * self.config.gateway_weight
        
        final_score = local_score + gateway_score
        
        if final_score >= 2.5:
            final = Decision.BLOCK
        elif final_score >= 1.5:
            final = Decision.CHALLENGE
        elif final_score >= 0.5:
            final = Decision.FLAG
        else:
            final = Decision.ALLOW
        
        # No consensus if decisions differ significantly
        consensus = abs(self._decision_score(local.decision) - self._decision_score(gateway.decision)) <= 1
        
        if not consensus:
            logger.warning(
                "No consensus: local=%s gateway=%s -> %s (escalating)",
                local.decision.value, gateway.decision.value, final.value
            )
        
        return final, consensus
    
    def _decision_score(self, decision: Decision) -> float:
        scores = {
            Decision.ALLOW: 0.0,
            Decision.FLAG: 1.0,
            Decision.CHALLENGE: 2.0,
            Decision.BLOCK: 3.0,
        }
        return scores.get(decision, 0.0)


class AresSecurityMiddleware:
    """
    Embedded security middleware for application-layer defense.
    
    Usage:
        middleware = AresSecurityMiddleware(config)
        
        # In your request handler:
        decision = await middleware.inspect(user_prompt, context)
        
        if decision.decision == Decision.BLOCK:
            raise SecurityException("Request blocked by ARES")
        elif decision.decision == Decision.CHALLENGE:
            # Require additional verification
            await require_mfa()
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.detector = StructuralAttackDetector(
            enabled_categories=self.config.enabled_categories,
            severity_threshold=self.config.severity_threshold,
        )
        self.cache = LRUCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )
        self.gateway = GatewayClient(self.config)
        self.consensus = ConsensusEngine(self.config)
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "local_blocks": 0,
            "gateway_calls": 0,
            "consensus_failures": 0,
            "errors": 0,
        }
    
    async def inspect(
        self,
        prompt: str,
        context: ConversationContext,
    ) -> SecurityDecision:
        """
        Inspect a prompt and return security decision.
        
        This is the main entry point - call for each user message.
        """
        start_time = time.perf_counter()
        correlation_id = f"ars-{context.conversation_id}-{context.turn_number}"
        
        self._stats["total_requests"] += 1
        
        # 1. Check cache
        cached = self.cache.get(prompt, context)
        if cached is not None:
            self._stats["cache_hits"] += 1
            latency = (time.perf_counter() - start_time) * 1000
            return SecurityDecision(
                decision=cached.decision,
                local_result=cached,
                cache_hit=True,
                correlation_id=correlation_id,
                latency_ms=latency,
            )
        
        # 2. Local structural detection (always runs, <1ms)
        local_result = self.detector.detect(prompt)
        
        # 3. Fast path: definitive local decision
        if local_result.decision in (Decision.ALLOW, Decision.BLOCK) and not self.config.require_consensus:
            self._maybe_cache(prompt, local_result, context)
            self._update_stats(local_result.decision)
            latency = (time.perf_counter() - start_time) * 1000
            return SecurityDecision(
                decision=local_result.decision,
                local_result=local_result,
                correlation_id=correlation_id,
                latency_ms=latency,
            )
        
        # 4. Async gateway call (non-blocking if configured)
        gateway_result = None
        if self.config.gateway_enabled:
            if self.config.async_gateway:
                # Fire and forget for ALLOW/FLAG, await for CHALLENGE/BLOCK
                if local_result.decision in (Decision.CHALLENGE, Decision.BLOCK):
                    gateway_result = await self.gateway.enforce(prompt, context)
                    self._stats["gateway_calls"] += 1
                else:
                    # Schedule background check
                    asyncio.create_task(self._background_gateway_check(prompt, context))
            else:
                gateway_result = await self.gateway.enforce(prompt, context)
                self._stats["gateway_calls"] += 1
        
        # 5. Consensus
        final_decision, consensus = self.consensus.decide(local_result, gateway_result)
        
        if not consensus:
            self._stats["consensus_failures"] += 1
            # Default to more restrictive
            final_decision = max(local_result.decision, gateway_result.decision if gateway_result else Decision.ALLOW)
        
        # 6. Cache result
        final_result = DetectionResult()
        for d in local_result.detections:
            final_result.add_detection(d)
        if gateway_result:
            for d in gateway_result.detections:
                final_result.add_detection(d)
        final_result.decision = final_decision
        
        self._maybe_cache(prompt, final_result, context)
        self._update_stats(final_decision)
        
        latency = (time.perf_counter() - start_time) * 1000
        
        if self.config.log_decisions:
            logger.log(self.config.log_level,
                "ARES decision: %s | local=%s gateway=%s consensus=%s latency=%.2fms",
                final_decision.value, local_result.decision.value,
                gateway_result.decision.value if gateway_result else "N/A",
                consensus, latency
            )
        
        return SecurityDecision(
            decision=final_decision,
            local_result=local_result,
            gateway_result=gateway_result,
            consensus=consensus,
            latency_ms=latency,
            correlation_id=correlation_id,
        )
    
    async def _background_gateway_check(self, prompt: str, context: ConversationContext) -> None:
        """Background gateway check for async mode."""
        try:
            await self.gateway.enforce(prompt, context)
        except Exception as e:
            logger.debug("Background gateway check failed: %s", e)
    
    def _maybe_cache(self, prompt: str, result: DetectionResult, context: ConversationContext) -> None:
        """Cache result based on policy."""
        if self.config.cache_policy == CachePolicy.NONE:
            return
        if self.config.cache_policy == CachePolicy.ALLOW and result.decision != Decision.ALLOW:
            return
        if self.config.cache_policy == CachePolicy.DENY and result.decision == Decision.ALLOW:
            return
        self.cache.set(prompt, result, context)
    
    def _update_stats(self, decision: Decision) -> None:
        if decision == Decision.BLOCK:
            self._stats["local_blocks"] += 1
    
    def get_stats(self) -> dict:
        """Get middleware statistics."""
        cache_stats = self.cache.stats()
        return {
            **self._stats,
            "cache": cache_stats,
            "cache_hit_rate": self._stats["cache_hits"] / max(self._stats["total_requests"], 1),
        }
    
    async def close(self):
        """Cleanup resources."""
        await self.gateway.close()
        self.cache.clear()


# FastAPI/Starlette integration
class AresFastAPIMiddleware:
    """FastAPI middleware for automatic request inspection."""
    
    def __init__(self, app, config: Optional[SecurityConfig] = None):
        self.app = app
        self.middleware = AresSecurityMiddleware(config)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Only inspect POST requests with JSON body to chat endpoints
        if scope["method"] != "POST" or not scope["path"].startswith(("/chat", "/v1/chat", "/api/chat")):
            await self.app(scope, receive, send)
            return
        
        # Read body
        body = b""
        async for message in receive:
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
        
        try:
            import json
            data = json.loads(body)
            prompt = data.get("prompt") or data.get("message", "")
            if prompt:
                context = ConversationContext(
                    conversation_id=scope.get("headers", {}).get(b"x-conversation-id", b"unknown").decode(),
                    turn_number=1,
                    application_name=scope.get("headers", {}).get(b"x-application-name", b"default").decode(),
                )
                decision = await self.middleware.inspect(prompt, context)
                
                if decision.decision == Decision.BLOCK:
                    # Return 403
                    await send({
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"application/json")],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": b'{"error": "Request blocked by ARES security policy"}',
                    })
                    return
                elif decision.decision == Decision.CHALLENGE:
                    # Add header for upstream handling
                    scope["ares_decision"] = decision
        
        except Exception as e:
            logger.debug("ARES middleware inspection failed: %s", e)
        
        # Replay body to app
        async def receive_wrapper():
            yield {"type": "http.request", "body": body, "more_body": False}
        
        await self.app(scope, receive_wrapper().__aiter__(), send)


# Convenience function for simple usage
async def quick_inspect(prompt: str, config: Optional[SecurityConfig] = None) -> SecurityDecision:
    """Quick inspection with default settings."""
    middleware = AresSecurityMiddleware(config)
    context = ConversationContext(conversation_id="quick", turn_number=1)
    return await middleware.inspect(prompt, context)