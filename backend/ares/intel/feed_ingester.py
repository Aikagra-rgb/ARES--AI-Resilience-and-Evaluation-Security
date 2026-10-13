"""
Verified Threat Intelligence Ingestion Pipeline.

Safely parses and normalizes external security advisories (MITRE ATLAS, OWASP LLM Top 10,
and custom security feeds), applies strict sanitization, and queues candidate samples into
the Attack Moderation Queue for human verification.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any, Dict, List, Optional

from ares.intel.models import AdvisorySource, IngestResult, ThreatIntelAdvisory
from ares.moderation.queue import AttackModerationQueue

logger = logging.getLogger(__name__)


class ThreatIntelIngesterError(Exception):
    """Exception raised for threat intel ingestion failures."""
    pass


class ThreatIntelIngester:
    """
    Ingests and normalizes security intelligence feeds, sanitizes adversarial payloads,
    and submits samples to the Attack Moderation Queue.
    """

    def __init__(
        self,
        moderation_queue: AttackModerationQueue,
        max_samples_per_feed: int = 100,
        max_prompt_length: int = 16384,
    ):
        self.moderation_queue = moderation_queue
        self.max_samples_per_feed = max_samples_per_feed
        self.max_prompt_length = max_prompt_length

    def sanitize_prompt(self, raw_prompt: str) -> Optional[str]:
        """
        Sanitize raw prompt text:
        - Strip null bytes and non-printable control characters
        - Normalize unicode to NFKC
        - Enforce length limits
        """
        if not raw_prompt or not isinstance(raw_prompt, str):
            return None

        # Remove null bytes
        cleaned = raw_prompt.replace("\x00", "")

        # Normalize unicode
        cleaned = unicodedata.normalize("NFKC", cleaned)

        # Strip whitespace
        cleaned = cleaned.strip()

        if not cleaned:
            return None

        # Length enforcement
        if len(cleaned) > self.max_prompt_length:
            cleaned = cleaned[: self.max_prompt_length]

        return cleaned

    def parse_mitre_atlas(
        self, feed_data: Any
    ) -> List[ThreatIntelAdvisory]:
        """
        Parse MITRE ATLAS techniques or case study exports.
        Accepts dictionary or list format.
        """
        advisories: List[ThreatIntelAdvisory] = []
        raw_items = feed_data.get("techniques", []) if isinstance(feed_data, dict) else feed_data

        if not isinstance(raw_items, list):
            raise ThreatIntelIngesterError("MITRE ATLAS feed data must contain a list of techniques")

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            technique_id = item.get("id") or item.get("technique_id")
            title = item.get("name") or item.get("title") or "Unknown MITRE Technique"
            description = item.get("description", "")
            severity = item.get("severity", "high").lower()

            # Map category from tactic or default
            tactic = item.get("tactic", "").lower()
            category = "jailbreak"
            if "evasion" in tactic or "evade" in title.lower():
                category = "evasion"
            elif "injection" in title.lower() or "injection" in description.lower():
                category = "injection"
            elif "exfiltration" in tactic or "extract" in title.lower():
                category = "extraction"

            prompts = item.get("prompts") or item.get("sample_prompts") or item.get("examples") or []
            if isinstance(prompts, str):
                prompts = [prompts]

            references = item.get("references", [])
            if isinstance(references, str):
                references = [references]

            advisories.append(
                ThreatIntelAdvisory(
                    source=AdvisorySource.MITRE_ATLAS.value,
                    technique_id=technique_id,
                    title=title,
                    description=description,
                    category=category,
                    severity=severity,
                    sample_prompts=prompts,
                    references=references,
                    metadata={"tactic": item.get("tactic"), "matrix": "ATLAS"},
                )
            )

        return advisories

    def parse_owasp_llm(
        self, feed_data: Any
    ) -> List[ThreatIntelAdvisory]:
        """
        Parse OWASP Top 10 for LLM Applications feed.
        """
        advisories: List[ThreatIntelAdvisory] = []
        raw_items = feed_data.get("vulnerabilities", []) if isinstance(feed_data, dict) else feed_data

        if not isinstance(raw_items, list):
            raise ThreatIntelIngesterError("OWASP LLM feed data must contain a list of entries")

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            vuln_id = item.get("id") or item.get("vuln_id")
            title = item.get("title") or item.get("name") or "OWASP LLM Vulnerability"
            description = item.get("description", "")
            severity = item.get("severity", "critical").lower()

            # Map category
            category = "jailbreak"
            if "llm01" in str(vuln_id).lower() or "prompt injection" in title.lower():
                category = "injection"
            elif "llm06" in str(vuln_id).lower() or "leak" in title.lower():
                category = "extraction"
            elif "dos" in str(vuln_id).lower() or "denial" in title.lower():
                category = "dos"

            prompts = item.get("examples") or item.get("sample_prompts") or item.get("prompts") or []
            if isinstance(prompts, str):
                prompts = [prompts]

            references = item.get("references", [])
            if isinstance(references, str):
                references = [references]

            advisories.append(
                ThreatIntelAdvisory(
                    source=AdvisorySource.OWASP_LLM.value,
                    technique_id=vuln_id,
                    title=title,
                    description=description,
                    category=category,
                    severity=severity,
                    sample_prompts=prompts,
                    references=references,
                    metadata={"project": "OWASP Top 10 for LLM"},
                )
            )

        return advisories

    def parse_generic_feed(
        self, feed_data: Any
    ) -> List[ThreatIntelAdvisory]:
        """
        Parse a generic JSON list of threat advisories.
        """
        advisories: List[ThreatIntelAdvisory] = []
        raw_items = feed_data.get("advisories", []) if isinstance(feed_data, dict) else feed_data

        if not isinstance(raw_items, list):
            raise ThreatIntelIngesterError("Generic feed data must be a list of advisory items")

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            prompts = item.get("sample_prompts") or item.get("prompts") or item.get("examples") or []
            if isinstance(prompts, str):
                prompts = [prompts]

            advisories.append(
                ThreatIntelAdvisory(
                    source=item.get("source", AdvisorySource.CUSTOM_FEED.value),
                    technique_id=item.get("technique_id") or item.get("id"),
                    title=item.get("title", "Advisory Entry"),
                    description=item.get("description", ""),
                    category=item.get("category", "jailbreak"),
                    severity=item.get("severity", "medium").lower(),
                    sample_prompts=prompts,
                    references=item.get("references", []),
                    metadata=item.get("metadata", {}),
                )
            )

        return advisories

    async def ingest_feed(
        self,
        feed_type: str,
        data: Any,
        feed_name: str = "threat_intel",
    ) -> IngestResult:
        """
        Parse, sanitize, and queue sample prompts from the specified threat intelligence feed.
        """
        feed_type_lower = feed_type.lower()
        if feed_type_lower in ("mitre_atlas", "atlas", "mitre"):
            advisories = self.parse_mitre_atlas(data)
        elif feed_type_lower in ("owasp_llm", "owasp"):
            advisories = self.parse_owasp_llm(data)
        elif feed_type_lower in ("generic", "custom", "json"):
            advisories = self.parse_generic_feed(data)
        else:
            raise ValueError(f"Unsupported feed type '{feed_type}'")

        queued_count = 0
        skipped_count = 0
        errors: List[str] = []

        for advisory in advisories:
            for raw_prompt in advisory.sample_prompts:
                # Check rate limit quota for single feed run
                if queued_count >= self.max_samples_per_feed:
                    skipped_count += 1
                    continue

                sanitized = self.sanitize_prompt(raw_prompt)
                if not sanitized:
                    skipped_count += 1
                    continue

                try:
                    meta = {
                        "technique_id": advisory.technique_id,
                        "advisory_title": advisory.title,
                        "feed_name": feed_name,
                        "references": advisory.references,
                    }
                    meta.update(advisory.metadata)

                    await self.moderation_queue.submit(
                        attack_text=sanitized,
                        category=advisory.category,
                        severity=advisory.severity,
                        source=f"threat_intel:{feed_name}:{advisory.source}",
                        domain="threat_intel",
                        target_prompt_hash=advisory.technique_id or "threat_intel_sample",
                        metadata=meta,
                        bypass_moderation=False,  # Always queue for verification
                    )
                    queued_count += 1
                except Exception as exc:
                    logger.warning("Failed to queue prompt from advisory '%s': %s", advisory.title, exc)
                    errors.append(f"Prompt from '{advisory.title}' failed: {exc}")
                    skipped_count += 1

        return IngestResult(
            feed_name=feed_name,
            advisories_count=len(advisories),
            prompts_queued=queued_count,
            prompts_skipped=skipped_count,
            errors=errors,
        )
