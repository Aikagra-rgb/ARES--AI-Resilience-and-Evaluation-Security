"""
Data models for the Attack Moderation Queue and Poisoning Defense.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ModerationStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class SubmissionSeverity(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


@dataclass
class ModerationItem:
    """Represents an adversarial sample queued for moderation and poisoning inspection."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attack_text: str = ""
    category: str = "jailbreak"
    severity: str = SubmissionSeverity.HIGH.value
    source: str = "redteam_probe"
    status: ModerationStatus = ModerationStatus.PENDING
    domain: str = "general"
    target_prompt_hash: str = "moderated_sample"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Submitter and Reviewer Tracking
    submitted_by: Optional[str] = None
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    reviewer_notes: Optional[str] = None

    # Poisoning & Quality Defense Metrics
    deduplication_score: Optional[float] = None
    duplicate_of_id: Optional[str] = None
    poisoning_flags: List[str] = field(default_factory=list)
    indexed_point_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary representation."""
        return {
            "id": self.id,
            "attack_text": self.attack_text,
            "category": self.category,
            "severity": self.severity,
            "source": self.source,
            "status": (
                self.status.value
                if isinstance(self.status, ModerationStatus)
                else self.status
            ),
            "domain": self.domain,
            "target_prompt_hash": self.target_prompt_hash,
            "metadata": self.metadata,
            "submitted_by": self.submitted_by,
            "submitted_at": self.submitted_at,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "reviewer_notes": self.reviewer_notes,
            "deduplication_score": self.deduplication_score,
            "duplicate_of_id": self.duplicate_of_id,
            "poisoning_flags": list(self.poisoning_flags),
            "indexed_point_id": self.indexed_point_id,
        }
