"""
Data models for the Threat Intelligence Ingestion Pipeline.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AdvisorySource(str, enum.Enum):
    MITRE_ATLAS = "mitre_atlas"
    OWASP_LLM = "owasp_llm"
    NIST_AI_RMF = "nist_ai_rmf"
    CUSTOM_FEED = "custom_feed"


@dataclass
class ThreatIntelAdvisory:
    """Standardized representation of an external security advisory."""

    advisory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = AdvisorySource.CUSTOM_FEED.value
    technique_id: Optional[str] = None
    title: str = ""
    description: str = ""
    category: str = "jailbreak"
    severity: str = "high"
    sample_prompts: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "advisory_id": self.advisory_id,
            "source": self.source,
            "technique_id": self.technique_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "sample_prompts": self.sample_prompts,
            "references": self.references,
            "metadata": self.metadata,
            "ingested_at": self.ingested_at,
        }


@dataclass
class IngestResult:
    """Outcome of a threat intel feed ingestion run."""

    feed_name: str
    advisories_count: int
    prompts_queued: int
    prompts_skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feed_name": self.feed_name,
            "advisories_count": self.advisories_count,
            "prompts_queued": self.prompts_queued,
            "prompts_skipped": self.prompts_skipped,
            "errors": self.errors,
        }
