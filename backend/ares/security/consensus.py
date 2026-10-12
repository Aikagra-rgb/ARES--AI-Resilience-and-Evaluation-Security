"""
ARES Consensus Engine.

Multi-layer security decision making with consensus protocols.
Combines deterministic, probabilistic, and human decisions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable

from ares.security.structural_detector import (
    DetectionResult,
    Decision,
    SeverityLevel,
    AttackType,
)

logger = logging.getLogger(__name__)


class DecisionSource(str, Enum):
    """Source of a security decision."""
    STRUCTURAL = "structural"           # Deterministic pattern matching
    VECTOR = "vector"                   # Semantic similarity
    ML_CLASSIFIER = "ml_classifier"     # ML model
    LLM_JUDGE = "llm_judge"             # LLM evaluation
    GATEWAY = "gateway"                 # Full gateway pipeline
    HUMAN = "human"                     # Human reviewer
    POLICY = "policy"                   # Static policy rules
    CONSENSUS = "consensus"             # Aggregated decision


@dataclass
class LayerDecision:
    """Decision from a single security layer."""
    source: DecisionSource
    decision: Decision
    confidence: float  # 0.0 - 1.0
    detections: List[Any] = field(default_factory=list)  # Detection objects
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 3),
            "detection_count": len(self.detections),
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class ConsensusResult:
    """Result of consensus protocol."""
    final_decision: Decision
    consensus_reached: bool
    participating_layers: List[DecisionSource]
    layer_decisions: List[LayerDecision]
    weights: Dict[DecisionSource, float]
    conflict_details: Optional[str] = None
    escalation_required: bool = False
    
    def to_dict(self) -> dict:
        return {
            "final_decision": self.final_decision.value,
            "consensus_reached": self.consensus_reached,
            "participating_layers": [s.value for s in self.participating_layers],
            "layer_decisions": [d.to_dict() for d in self.layer_decisions],
            "weights": {k.value: v for k, v in self.weights.items()},
            "conflict_details": self.conflict_details,
            "escalation_required": self.escalation_required,
        }


class ConsensusProtocol(ABC):
    """Abstract consensus protocol."""
    
    @abstractmethod
    async def decide(self, layer_decisions: List[LayerDecision]) -> ConsensusResult:
        pass


class WeightedVotingProtocol(ConsensusProtocol):
    """
    Weighted voting consensus.
    
    Each layer has a weight. Decision scores:
    - ALLOW = 0
    - FLAG = 1
    - CHALLENGE = 2
    - BLOCK = 3
    
    Weighted average determines final decision.
    """
    
    DECISION_SCORES = {
        Decision.ALLOW: 0.0,
        Decision.FLAG: 1.0,
        Decision.CHALLENGE: 2.0,
        Decision.BLOCK: 3.0,
    }
    
    DEFAULT_WEIGHTS = {
        DecisionSource.STRUCTURAL: 0.30,    # Deterministic, fast, high precision
        DecisionSource.VECTOR: 0.20,        # Semantic, catches variants
        DecisionSource.ML_CLASSIFIER: 0.15, # Statistical, generalizes
        DecisionSource.LLM_JUDGE: 0.20,     # Semantic understanding
        DecisionSource.GATEWAY: 0.25,       # Full pipeline
        DecisionSource.HUMAN: 1.00,         # Ultimate authority
        DecisionSource.POLICY: 0.50,        # Hard rules
    }
    
    def __init__(
        self,
        weights: Optional[Dict[DecisionSource, float]] = None,
        consensus_threshold: float = 0.6,  # Fraction of max score for consensus
        block_threshold: float = 2.0,
        challenge_threshold: float = 1.0,
        flag_threshold: float = 0.5,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.consensus_threshold = consensus_threshold
        self.block_threshold = block_threshold
        self.challenge_threshold = challenge_threshold
        self.flag_threshold = flag_threshold
    
    async def decide(self, layer_decisions: List[LayerDecision]) -> ConsensusResult:
        if not layer_decisions:
            return ConsensusResult(
                final_decision=Decision.ALLOW,
                consensus_reached=True,
                participating_layers=[],
                layer_decisions=[],
                weights={},
            )
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        participating = []
        
        for ld in layer_decisions:
            weight = self.weights.get(ld.source, 0.1)
            score = self.DECISION_SCORES.get(ld.decision, 0.0)
            weighted_score += score * weight * ld.confidence
            total_weight += weight
            participating.append(ld.source)
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine final decision
        if avg_score >= self.block_threshold:
            final = Decision.BLOCK
        elif avg_score >= self.challenge_threshold:
            final = Decision.CHALLENGE
        elif avg_score >= self.flag_threshold:
            final = Decision.FLAG
        else:
            final = Decision.ALLOW
        
        # Check consensus: all non-zero-weight layers agree within 1 level
        decisions = [ld.decision for ld in layer_decisions if self.weights.get(ld.source, 0) > 0]
        if decisions:
            scores = [self.DECISION_SCORES[d] for d in decisions]
            consensus_reached = (max(scores) - min(scores)) <= 1.0
        else:
            consensus_reached = True
        
        # Conflict details
        conflict = None
        if not consensus_reached:
            conflict = f"Layer disagreement: {[f'{ld.source.value}={ld.decision.value}' for ld in layer_decisions]}"
        
        # Escalation if high-confidence conflict
        escalation = False
        if not consensus_reached:
            high_conf_conflicts = [
                ld for ld in layer_decisions
                if ld.confidence >= 0.8 and ld.decision in (Decision.CHALLENGE, Decision.BLOCK)
            ]
            if len(high_conf_conflicts) >= 2:
                escalation = True
        
        return ConsensusResult(
            final_decision=final,
            consensus_reached=consensus_reached,
            participating_layers=participating,
            layer_decisions=layer_decisions,
            weights=self.weights,
            conflict_details=conflict,
            escalation_required=escalation,
        )


class UnanimousProtocol(ConsensusProtocol):
    """Require unanimous agreement for restrictive decisions."""
    
    def __init__(self, restrictive_decisions: List[Decision] = None):
        self.restrictive = restrictive_decisions or [Decision.CHALLENGE, Decision.BLOCK]
    
    async def decide(self, layer_decisions: List[LayerDecision]) -> ConsensusResult:
        if not layer_decisions:
            return ConsensusResult(
                final_decision=Decision.ALLOW,
                consensus_reached=True,
                participating_layers=[],
                layer_decisions=[],
                weights={},
            )
        
        decisions = [ld.decision for ld in layer_decisions]
        
        # If any layer says BLOCK, and not unanimous, escalate
        if Decision.BLOCK in decisions:
            if all(d == Decision.BLOCK for d in decisions):
                return ConsensusResult(
                    final_decision=Decision.BLOCK,
                    consensus_reached=True,
                    participating_layers=[ld.source for ld in layer_decisions],
                    layer_decisions=layer_decisions,
                    weights={},
                )
            else:
                return ConsensusResult(
                    final_decision=Decision.CHALLENGE,  # Escalate
                    consensus_reached=False,
                    participating_layers=[ld.source for ld in layer_decisions],
                    layer_decisions=layer_decisions,
                    weights={},
                    conflict_details="Split decision on BLOCK",
                    escalation_required=True,
                )
        
        # If any layer says CHALLENGE
        if Decision.CHALLENGE in decisions:
            if all(d in (Decision.CHALLENGE, Decision.FLAG) for d in decisions):
                return ConsensusResult(
                    final_decision=Decision.CHALLENGE,
                    consensus_reached=True,
                    participating_layers=[ld.source for ld in layer_decisions],
                    layer_decisions=layer_decisions,
                    weights={},
                )
            else:
                return ConsensusResult(
                    final_decision=Decision.FLAG,
                    consensus_reached=False,
                    participating_layers=[ld.source for ld in layer_decisions],
                    layer_decisions=layer_decisions,
                    weights={},
                    conflict_details="Split decision on CHALLENGE",
                )
        
        # Default: majority vote
        return await WeightedVotingProtocol().decide(layer_decisions)


class HierarchicalProtocol(ConsensusProtocol):
    """
    Hierarchical decision making.
    
    Priority order (highest first):
    1. HUMAN - Always wins
    2. POLICY - Hard rules
    3. STRUCTURAL - Deterministic patterns
    4. GATEWAY - Full pipeline
    5. VECTOR - Semantic
    6. ML_CLASSIFIER - Statistical
    7. LLM_JUDGE - Probabilistic
    
    Higher priority layers can override lower ones.
    """
    
    PRIORITY = [
        DecisionSource.HUMAN,
        DecisionSource.POLICY,
        DecisionSource.STRUCTURAL,
        DecisionSource.GATEWAY,
        DecisionSource.VECTOR,
        DecisionSource.ML_CLASSIFIER,
        DecisionSource.LLM_JUDGE,
    ]
    
    async def decide(self, layer_decisions: List[LayerDecision]) -> ConsensusResult:
        if not layer_decisions:
            return ConsensusResult(
                final_decision=Decision.ALLOW,
                consensus_reached=True,
                participating_layers=[],
                layer_decisions=[],
                weights={},
            )
        
        # Sort by priority
        sorted_decisions = sorted(
            layer_decisions,
            key=lambda ld: self.PRIORITY.index(ld.source) if ld.source in self.PRIORITY else 999
        )
        
        # Highest priority non-ALLOW wins
        for ld in sorted_decisions:
            if ld.decision != Decision.ALLOW:
                return ConsensusResult(
                    final_decision=ld.decision,
                    consensus_reached=True,
                    participating_layers=[d.source for d in layer_decisions],
                    layer_decisions=layer_decisions,
                    weights={},
                    conflict_details=f"Overridden by {ld.source.value}",
                )
        
        # All ALLOW
        return ConsensusResult(
            final_decision=Decision.ALLOW,
            consensus_reached=True,
            participating_layers=[d.source for d in layer_decisions],
            layer_decisions=layer_decisions,
            weights={},
        )


class ConsensusEngine:
    """
    Main consensus engine with multiple protocol support.
    
    Features:
    - Multiple consensus protocols
    - Automatic protocol selection
    - Conflict logging and alerting
    - Escalation handling
    """
    
    def __init__(
        self,
        protocol: str = "weighted",
        custom_weights: Optional[Dict[DecisionSource, float]] = None,
        escalation_handler: Optional[Callable[[ConsensusResult], Awaitable[None]]] = None,
    ):
        self.escalation_handler = escalation_handler
        
        if protocol == "weighted":
            self.protocol = WeightedVotingProtocol(weights=custom_weights)
        elif protocol == "unanimous":
            self.protocol = UnanimousProtocol()
        elif protocol == "hierarchical":
            self.protocol = HierarchicalProtocol()
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    
    async def decide(
        self,
        layer_decisions: List[LayerDecision],
    ) -> ConsensusResult:
        """Run consensus protocol on layer decisions."""
        result = await self.protocol.decide(layer_decisions)
        
        # Handle escalation
        if result.escalation_required and self.escalation_handler:
            try:
                await self.escalation_handler(result)
            except Exception as e:
                logger.error("Escalation handler failed: %s", e)
        
        # Log conflicts
        if not result.consensus_reached:
            logger.warning(
                "Consensus conflict: %s | final=%s | layers=%s",
                result.conflict_details,
                result.final_decision.value,
                [s.value for s in result.participating_layers]
            )
        
        return result
    
    @staticmethod
    def create_layer_decision(
        source: DecisionSource,
        detection_result: DetectionResult,
        latency_ms: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> LayerDecision:
        """Create LayerDecision from DetectionResult."""
        return LayerDecision(
            source=source,
            decision=detection_result.decision,
            confidence=detection_result.total_confidence,
            detections=detection_result.detections,
            metadata=metadata or {},
            latency_ms=latency_ms,
        )


# Async escalation handlers
async def log_escalation(result: ConsensusResult) -> None:
    """Log escalation to security team."""
    logger.critical(
        "SECURITY ESCALATION: %s | conflict=%s | decision=%s",
        result.conflict_details,
        result.final_decision.value,
        [s.value for s in result.participating_layers]
    )


async def webhook_escalation(result: ConsensusResult, webhook_url: str) -> None:
    """Send escalation to webhook (Slack, PagerDuty, etc.)."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json={
                "alert": "ARES Security Escalation",
                "decision": result.final_decision.value,
                "conflict": result.conflict_details,
                "layers": [s.value for s in result.participating_layers],
                "details": [d.to_dict() for d in result.layer_decisions],
            })
    except Exception as e:
        logger.error("Webhook escalation failed: %s", e)


async def email_escalation(result: ConsensusResult, recipients: List[str]) -> None:
    """Send escalation email."""
    # Implementation would use SMTP
    logger.info("Would email escalation to %s: %s", recipients, result.conflict_details)