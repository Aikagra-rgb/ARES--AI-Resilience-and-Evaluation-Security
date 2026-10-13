"""
Attack Moderation Queue and Dataset Poisoning Defense Engine.

Prevents malicious or low-quality adversarial prompts from directly polluting
production vector indexes (Qdrant) or training corpora.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ares.moderation.models import ModerationItem, ModerationStatus, SubmissionSeverity
from ares.redteam.attacker import AttackAttempt
from ares.vectordb.store import QdrantStore

logger = logging.getLogger(__name__)

# Heuristic list of common benign phrases to detect label-inversion poisoning attacks
_BENIGN_PATTERNS = [
    r"^hello(?:\s+there|\s+world)?[\.!\?]*$",
    r"^hi(?:\s+there)?[\.!\?]*$",
    r"^how are you(?: today)?[\.!\?]*$",
    r"^what is the weather(?: like)?[\.!\?]*$",
    r"^good (?:morning|afternoon|evening)[\.!\?]*$",
    r"^thank you(?: very much)?[\.!\?]*$",
    r"^tell me a (?:joke|story|recipe)[\.!\?]*$",
    r"^what is (?:the capital of|2\s*\+\s*2)[\.!\?]*$",
]
_COMPILED_BENIGN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _BENIGN_PATTERNS]


class ModerationQueueError(Exception):
    """Exception raised for moderation queue errors."""
    pass


class AttackModerationQueue:
    """
    Manages pending adversarial samples, executes automated poisoning & deduplication
    evaluations, and facilitates human-in-the-loop approval before vector DB upsertion.
    """

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        dedup_threshold: float = 0.92,
        max_text_length: int = 32768,
        auto_reject_poisoned: bool = False,
    ):
        self.qdrant_store = qdrant_store
        self.dedup_threshold = dedup_threshold
        self.max_text_length = max_text_length
        self.auto_reject_poisoned = auto_reject_poisoned

        self._items: Dict[str, ModerationItem] = {}
        self._lock = asyncio.Lock()

    def _inspect_poisoning_signals(
        self,
        attack_text: str,
        category: str,
        severity: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """Detect automated poisoning flags in submitted payload."""
        flags: List[str] = []
        stripped = attack_text.strip()

        # 1. Null-byte or binary control char injection
        if "\x00" in attack_text:
            flags.append("null_byte_injection")

        # 2. Text length boundary
        if len(attack_text) > self.max_text_length:
            flags.append("oversized_payload")

        # 3. Label-inversion poisoning: Benign conversational text marked as critical/high attack
        if severity in (SubmissionSeverity.CRITICAL.value, SubmissionSeverity.HIGH.value):
            for pat in _COMPILED_BENIGN_PATTERNS:
                if pat.match(stripped):
                    flags.append("label_inversion_detected")
                    break

        # 4. Low-entropy repetition (e.g., 'a' * 500)
        if len(stripped) > 50 and len(set(stripped)) < 5:
            flags.append("low_entropy_garbage")

        # 5. Metadata validation
        for key, val in metadata.items():
            if not isinstance(key, str) or len(key) > 128:
                flags.append("suspicious_metadata_key")
            if isinstance(val, str) and ("<script>" in val.lower() or "\x00" in val):
                flags.append("malicious_metadata_payload")

        return flags

    async def _check_semantic_deduplication(
        self, attack_text: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Query Qdrant store to detect if an identical or near-duplicate sample exists."""
        if not self.qdrant_store:
            return None, None

        try:
            hits = await self.qdrant_store.search_similar_attacks(
                query_text=attack_text,
                limit=1,
            )
            if hits and len(hits) > 0:
                top_hit = hits[0]
                score = float(top_hit.get("score", 0.0))
                point_id = str(top_hit.get("id"))
                if score >= self.dedup_threshold:
                    return score, point_id
                return score, None
        except Exception as exc:
            logger.warning("Failed to evaluate semantic deduplication in Qdrant: %s", exc)

        return None, None

    async def submit(
        self,
        attack_text: str,
        category: str = "jailbreak",
        severity: str = SubmissionSeverity.HIGH.value,
        source: str = "redteam_probe",
        submitted_by: Optional[str] = None,
        domain: str = "general",
        target_prompt_hash: str = "moderated_sample",
        metadata: Optional[Dict[str, Any]] = None,
        bypass_moderation: bool = False,
    ) -> ModerationItem:
        """
        Submit a candidate attack prompt to the moderation queue.
        Executes poisoning heuristic checks and semantic deduplication.
        """
        if not attack_text or not attack_text.strip():
            raise ValueError("attack_text cannot be empty or whitespace")

        meta = dict(metadata or {})
        poisoning_flags = self._inspect_poisoning_signals(attack_text, category, severity, meta)

        dedup_score, duplicate_of_id = await self._check_semantic_deduplication(attack_text)
        if duplicate_of_id:
            poisoning_flags.append("semantic_duplicate")

        item = ModerationItem(
            attack_text=attack_text,
            category=category,
            severity=severity,
            source=source,
            domain=domain,
            target_prompt_hash=target_prompt_hash,
            metadata=meta,
            submitted_by=submitted_by,
            deduplication_score=dedup_score,
            duplicate_of_id=duplicate_of_id,
            poisoning_flags=poisoning_flags,
        )

        async with self._lock:
            # Check auto-rejection on severe poisoning
            if self.auto_reject_poisoned and poisoning_flags:
                item.status = ModerationStatus.REJECTED
                item.reviewed_by = "automated_poisoning_defense"
                item.reviewed_at = datetime.now(timezone.utc).isoformat()
                item.reviewer_notes = f"Auto-rejected by poisoning defense flags: {', '.join(poisoning_flags)}"
            elif bypass_moderation and not poisoning_flags:
                # Immediate indexing if trusted and clean
                item.status = ModerationStatus.APPROVED
                item.reviewed_by = submitted_by or "system_trusted"
                item.reviewed_at = datetime.now(timezone.utc).isoformat()
                item.reviewer_notes = "Pre-approved clean sample"
                if self.qdrant_store:
                    await self._index_to_qdrant(item)

            self._items[item.id] = item

        return item

    async def _index_to_qdrant(self, item: ModerationItem) -> str:
        """Upsert approved attack item into Qdrant store as an AttackAttempt."""
        if not self.qdrant_store:
            raise ModerationQueueError("No QdrantStore configured for indexing")

        attempt = AttackAttempt(
            attempt_number=1,
            attack_text=item.attack_text,
            category=item.category,
            victim_response="Moderated adversarial sample",
            rule_verdict=True,
            llm_verdict=True,
            success=True,
            reasoning=f"Approved sample via moderation queue: {item.reviewer_notes or 'Verified'}",
            domain=item.domain,
            mutation_path=["moderated_ingest"],
            operator_applied=item.metadata.get("technique_id"),
            canary_tokens_found=[],
            attacker_provider=item.source,
            victim_provider="verified_corpus",
            judge_provider=item.reviewed_by or "moderation_queue",
        )

        # Set extended metadata tags on the attempt if present
        setattr(attempt, "severity", item.severity)
        setattr(attempt, "source", item.source)
        setattr(attempt, "tags", ["moderated_corpus", item.category, item.source])

        point_id = await self.qdrant_store.upsert_attempt(
            attempt=attempt,
            target_prompt_hash=item.target_prompt_hash,
        )
        item.indexed_point_id = point_id
        return point_id

    async def approve(
        self,
        item_id: str,
        reviewer_id: str,
        reviewer_notes: Optional[str] = None,
        qdrant_store: Optional[QdrantStore] = None,
    ) -> ModerationItem:
        """Approve a pending sample and index it into Qdrant."""
        async with self._lock:
            item = self._items.get(item_id)
            if not item:
                raise KeyError(f"Moderation item not found: {item_id}")

            if item.status != ModerationStatus.PENDING:
                raise ValueError(
                    f"Item {item_id} cannot be approved from status {item.status.value}"
                )

            item.status = ModerationStatus.APPROVED
            item.reviewed_by = reviewer_id
            item.reviewed_at = datetime.now(timezone.utc).isoformat()
            item.reviewer_notes = reviewer_notes or "Approved by moderator"

            store = qdrant_store or self.qdrant_store
            if store:
                old_store = self.qdrant_store
                self.qdrant_store = store
                try:
                    await self._index_to_qdrant(item)
                finally:
                    self.qdrant_store = old_store

            return item

    async def reject(
        self,
        item_id: str,
        reviewer_id: str,
        reason: str,
    ) -> ModerationItem:
        """Reject a pending sample."""
        if not reason or not reason.strip():
            raise ValueError("Rejection reason must be provided")

        async with self._lock:
            item = self._items.get(item_id)
            if not item:
                raise KeyError(f"Moderation item not found: {item_id}")

            if item.status != ModerationStatus.PENDING:
                raise ValueError(
                    f"Item {item_id} cannot be rejected from status {item.status.value}"
                )

            item.status = ModerationStatus.REJECTED
            item.reviewed_by = reviewer_id
            item.reviewed_at = datetime.now(timezone.utc).isoformat()
            item.reviewer_notes = reason.strip()
            return item

    async def get_item(self, item_id: str) -> Optional[ModerationItem]:
        """Retrieve a moderation item by its ID."""
        async with self._lock:
            return self._items.get(item_id)

    async def list_items(
        self,
        status: Optional[ModerationStatus] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ModerationItem]:
        """List queue items with optional filtering and pagination."""
        async with self._lock:
            items = list(self._items.values())

        if status is not None:
            items = [i for i in items if i.status == status]
        if category is not None:
            items = [i for i in items if i.category == category]

        # Sort newest first
        items.sort(key=lambda x: x.submitted_at, reverse=True)
        return items[offset : offset + limit]

    async def get_stats(self) -> Dict[str, Any]:
        """Return moderation queue statistics."""
        async with self._lock:
            items = list(self._items.values())

        total = len(items)
        pending = sum(1 for i in items if i.status == ModerationStatus.PENDING)
        approved = sum(1 for i in items if i.status == ModerationStatus.APPROVED)
        rejected = sum(1 for i in items if i.status == ModerationStatus.REJECTED)
        flagged = sum(1 for i in items if len(i.poisoning_flags) > 0)

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "flagged": flagged,
        }
