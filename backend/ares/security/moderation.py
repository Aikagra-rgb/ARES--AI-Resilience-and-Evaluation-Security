"""
ARES Attack Corpus Moderation System.

Provides manual approval workflow for indexing new attack vectors to prevent
ML poisoning and corpus pollution. Includes review queue, approval policies,
and audit trail.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Awaitable

from ares.config import Settings, settings as default_settings
from ares.redteam.attacker import AttackAttempt
from ares.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)


class ModerationStatus(str, enum.Enum):
    """Status of an attack sample in the moderation queue."""
    PENDING = "pending"           # Awaiting review
    APPROVED = "approved"         # Approved for indexing
    REJECTED = "rejected"         # Rejected (false positive, benign, etc.)
    ESCALATED = "escalated"       # Escalated to senior reviewer
    AUTO_APPROVED = "auto_approved"  # Auto-approved by policy


class ModerationDecision(str, enum.Enum):
    """Decision made by reviewer."""
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    REQUEST_MORE_INFO = "request_more_info"


@dataclass
class AttackSample:
    """An attack sample submitted for moderation."""
    attack_text: str
    category: str
    source: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Attribution
    discovered_by: str = "system"  # system, user, threat_intel
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Moderation
    status: ModerationStatus = ModerationStatus.PENDING
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    decision: Optional[ModerationDecision] = None
    review_notes: str = ""
    # Deduplication
    content_hash: str = field(init=False)
    similar_samples: List[str] = field(default_factory=list)  # Content hashes of similar samples
    
    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.attack_text.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "attack_text": self.attack_text,
            "category": self.category,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "discovered_by": self.discovered_by,
            "discovered_at": self.discovered_at.isoformat(),
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "decision": self.decision.value if self.decision else None,
            "review_notes": self.review_notes,
            "content_hash": self.content_hash,
            "similar_samples": self.similar_samples,
        }


@dataclass
class ModerationPolicy:
    """Policy for auto-approval/rejection of attack samples."""
    # Auto-approve if confidence >= threshold and no similar samples
    auto_approve_confidence_threshold: float = 0.95
    # Auto-reject if confidence <= threshold
    auto_reject_confidence_threshold: float = 0.3
    # Require human review for these categories regardless of confidence
    mandatory_review_categories: List[str] = field(default_factory=lambda: [
        "data_exfiltration",
        "tool_misuse",
        "encoding_or_obfuscation",
    ])
    # Max similar samples allowed for auto-approval
    max_similar_for_auto_approve: int = 0
    # Require minimum content length
    min_content_length: int = 20
    # Block list of known benign patterns
    benign_patterns: List[str] = field(default_factory=list)


@dataclass
class ReviewAction:
    """An action taken during review."""
    reviewer_id: str
    action: ModerationDecision
    notes: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AttackModerationQueue:
    """
    Moderation queue for attack samples with approval workflow.
    
    Features:
    - Auto-approval based on configurable policies
    - Human review queue with priority ordering
    - Similarity detection to prevent duplicate indexing
    - Audit trail of all moderation decisions
    - Escalation path for controversial samples
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantStore] = None,
        policy: Optional[ModerationPolicy] = None,
        notification_callback: Optional[Callable[[AttackSample], Awaitable[None]]] = None,
    ):
        self.settings = settings or default_settings
        self.store = store or QdrantStore(settings=self.settings)
        self.policy = policy or ModerationPolicy()
        self.notification_callback = notification_callback
        
        # In-memory queue (replace with persistent storage in production)
        self._queue: Dict[str, AttackSample] = {}
        self._reviews: Dict[str, List[ReviewAction]] = {}
        self._indexed_hashes: set = set()  # Track already indexed content hashes
    
    async def submit(self, sample: AttackSample) -> str:
        """
        Submit an attack sample for moderation.
        
        Returns:
            Sample ID (content hash)
        """
        # Check for duplicates
        if sample.content_hash in self._queue:
            existing = self._queue[sample.content_hash]
            existing.similar_samples.append(sample.content_hash)
            logger.info("Duplicate sample submitted: %s", sample.content_hash)
            return sample.content_hash
        
        # Check if already indexed
        if sample.content_hash in self._indexed_hashes:
            logger.info("Sample already indexed: %s", sample.content_hash)
            return sample.content_hash
        
        # Check similarity with existing samples
        similar = await self._find_similar_samples(sample)
        sample.similar_samples = similar
        
        # Apply auto-moderation policy
        await self._apply_auto_policy(sample)
        
        # Add to queue
        self._queue[sample.content_hash] = sample
        self._reviews[sample.content_hash] = []
        
        # Notify reviewers if needed
        if sample.status == ModerationStatus.PENDING and self.notification_callback:
            try:
                await self.notification_callback(sample)
            except Exception as e:
                logger.warning("Notification callback failed: %s", e)
        
        logger.info("Sample submitted for moderation: %s (status: %s)", 
                   sample.content_hash, sample.status.value)
        return sample.content_hash
    
    async def _apply_auto_policy(self, sample: AttackSample) -> None:
        """Apply automatic moderation policy."""
        # Check minimum content length
        if len(sample.attack_text) < self.policy.min_content_length:
            sample.status = ModerationStatus.REJECTED
            sample.decision = ModerationDecision.REJECT
            sample.review_notes = f"Content too short (< {self.policy.min_content_length} chars)"
            return
        
        # Check mandatory review categories
        if sample.category in self.policy.mandatory_review_categories:
            sample.status = ModerationStatus.PENDING
            return
        
        # Check confidence thresholds
        if sample.confidence >= self.policy.auto_approve_confidence_threshold:
            if len(sample.similar_samples) <= self.policy.max_similar_for_auto_approve:
                sample.status = ModerationStatus.AUTO_APPROVED
                sample.decision = ModerationDecision.APPROVE
                sample.review_notes = f"Auto-approved: confidence {sample.confidence:.2f} >= threshold"
                return
        
        if sample.confidence <= self.policy.auto_reject_confidence_threshold:
            sample.status = ModerationStatus.REJECTED
            sample.decision = ModerationDecision.REJECT
            sample.review_notes = f"Auto-rejected: confidence {sample.confidence:.2f} <= threshold"
            return
        
        # Default: pending human review
        sample.status = ModerationStatus.PENDING
    
    async def _find_similar_samples(self, sample: AttackSample) -> List[str]:
        """Find similar samples already in queue or indexed."""
        similar = []
        try:
            if self.store:
                hits = await self.store.search_similar_attacks(
                    query_text=sample.attack_text,
                    limit=5,
                    score_threshold=0.85,
                )
                similar = [h.get("payload", {}).get("content_hash", "") for h in hits if h.get("payload", {}).get("content_hash")]
        except Exception as e:
            logger.warning("Similarity search failed: %s", e)
        return similar
    
    async def review(
        self,
        sample_id: str,
        reviewer_id: str,
        decision: ModerationDecision,
        notes: str = "",
    ) -> bool:
        """
        Record a review decision.
        
        Returns:
            True if review was recorded, False if sample not found
        """
        if sample_id not in self._queue:
            return False
        
        sample = self._queue[sample_id]
        
        # Record review action
        action = ReviewAction(
            reviewer_id=reviewer_id,
            action=decision,
            notes=notes,
        )
        self._reviews[sample_id].append(action)
        
        # Update sample status
        if decision == ModerationDecision.APPROVE:
            sample.status = ModerationStatus.APPROVED
            sample.decision = decision
            sample.reviewed_at = datetime.now(timezone.utc)
            sample.reviewer_id = reviewer_id
            sample.review_notes = notes
            # Track as indexed
            self._indexed_hashes.add(sample_id)
        elif decision == ModerationDecision.REJECT:
            sample.status = ModerationStatus.REJECTED
            sample.decision = decision
            sample.reviewed_at = datetime.now(timezone.utc)
            sample.reviewer_id = reviewer_id
            sample.review_notes = notes
        elif decision == ModerationDecision.ESCALATE:
            sample.status = ModerationStatus.ESCALATED
            sample.review_notes = notes
        elif decision == ModerationDecision.REQUEST_MORE_INFO:
            sample.review_notes = notes
        
        logger.info("Sample %s reviewed by %s: %s", sample_id, reviewer_id, decision.value)
        return True
    
    async def get_pending(self, limit: int = 50, category: Optional[str] = None) -> List[AttackSample]:
        """Get pending samples for review."""
        pending = [
            s for s in self._queue.values()
            if s.status == ModerationStatus.PENDING
            and (category is None or s.category == category)
        ]
        # Sort by priority: higher confidence first, then older first
        pending.sort(key=lambda s: (-s.confidence, s.discovered_at))
        return pending[:limit]
    
    async def get_escalated(self, limit: int = 50) -> List[AttackSample]:
        """Get escalated samples."""
        escalated = [
            s for s in self._queue.values()
            if s.status == ModerationStatus.ESCALATED
        ]
        escalated.sort(key=lambda s: s.discovered_at)
        return escalated[:limit]
    
    async def get_sample(self, sample_id: str) -> Optional[AttackSample]:
        """Get a sample by ID."""
        return self._queue.get(sample_id)
    
    async def get_review_history(self, sample_id: str) -> List[ReviewAction]:
        """Get review history for a sample."""
        return self._reviews.get(sample_id, [])
    
    async def get_stats(self) -> dict:
        """Get moderation queue statistics."""
        total = len(self._queue)
        status_counts = {}
        for status in ModerationStatus:
            status_counts[status.value] = sum(1 for s in self._queue.values() if s.status == status)
        
        return {
            "total_samples": total,
            "pending_review": status_counts.get("pending", 0),
            "approved": status_counts.get("approved", 0) + status_counts.get("auto_approved", 0),
            "rejected": status_counts.get("rejected", 0),
            "escalated": status_counts.get("escalated", 0),
            "indexed": len(self._indexed_hashes),
            "queue_size": total,
        }
    
    async def index_approved(self, sample_id: str) -> bool:
        """
        Index an approved sample into the vector database.
        
        Returns:
            True if indexed successfully
        """
        if sample_id not in self._queue:
            return False
        
        sample = self._queue[sample_id]
        if sample.status not in (ModerationStatus.APPROVED, ModerationStatus.AUTO_APPROVED):
            return False
        
        try:
            # Create AttackAttempt from sample
            attempt = AttackAttempt(
                attempt_number=0,
                attack_text=sample.attack_text,
                category=sample.category,
                victim_response="",
                rule_verdict=False,
                llm_verdict=False,
                success=True,
                reasoning="Approved for indexing via moderation queue",
                domain="moderated",
                source=sample.source,
                attacker_provider="moderation_queue",
                victim_provider="unknown",
                judge_provider="moderation_queue",
            )
            
            # Index to Qdrant
            await self.store.upsert_attempt(
                attempt=attempt,
                target_prompt_hash=sample.content_hash,
            )
            
            self._indexed_hashes.add(sample_id)
            logger.info("Sample %s indexed to vector database", sample_id)
            return True
        except Exception as e:
            logger.error("Failed to index sample %s: %s", sample_id, e)
            return False


# Global moderation queue
_moderation_queue: Optional[AttackModerationQueue] = None


async def get_moderation_queue() -> AttackModerationQueue:
    """Get or create the global moderation queue."""
    global _moderation_queue
    if _moderation_queue is None:
        _moderation_queue = AttackModerationQueue()
    return _moderation_queue


async def submit_for_moderation(
    attack_text: str,
    category: str,
    source: str,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
    discovered_by: str = "system",
) -> str:
    """Convenience function to submit an attack sample for moderation."""
    queue = await get_moderation_queue()
    sample = AttackSample(
        attack_text=attack_text,
        category=category,
        source=source,
        confidence=confidence,
        metadata=metadata or {},
        discovered_by=discovered_by,
    )
    return await queue.submit(sample)