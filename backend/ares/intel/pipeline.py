"""
ARES Threat Intelligence Pipeline.

Continuous corpus updates from external threat intelligence sources.
Automates the process of fetching, validating, and indexing new attack patterns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, AsyncIterator
from urllib.parse import urlparse

import aiohttp

from ares.security.structural_detector import AttackType, Detection
from ares.vectordb.store import QdrantStore
from ares.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


@dataclass
class AttackSample:
    """A single attack sample from threat intelligence."""
    attack_text: str
    category: AttackType
    source: str
    source_url: str
    discovered_at: datetime
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Validation
    validated: bool = False
    validated_by: Optional[str] = None
    validated_at: Optional[datetime] = None
    false_positive: bool = False
    
    def to_dict(self) -> dict:
        return {
            "attack_text": self.attack_text,
            "category": self.category.value,
            "source": self.source,
            "source_url": self.source_url,
            "discovered_at": self.discovered_at.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata,
            "validated": self.validated,
            "validated_by": self.validated_by,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "false_positive": self.false_positive,
        }
    
    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(self.attack_text.encode()).hexdigest()[:16]


@dataclass
class ThreatSource:
    """Configuration for a threat intelligence source."""
    name: str
    url: str
    parser: str  # Parser class name
    enabled: bool = True
    schedule_cron: str = "0 */6 * * *"  # Every 6 hours
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit: float = 1.0  # Requests per second
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatIntelSource(ABC):
    """Abstract base class for threat intelligence sources."""
    
    def __init__(self, source: ThreatSource, session: aiohttp.ClientSession):
        self.source = source
        self.session = session
        self.last_fetch = 0.0
    
    @abstractmethod
    async def fetch(self) -> AsyncIterator[AttackSample]:
        """Fetch attack samples from source."""
        pass
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_fetch
        min_interval = 1.0 / self.source.rate_limit
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self.last_fetch = time.time()
    
    def _categorize_attack(self, text: str, metadata: Dict) -> AttackType:
        """Categorize attack based on content and metadata."""
        text_lower = text.lower()
        
        # Check metadata first
        if "category" in metadata:
            try:
                return AttackType(metadata["category"])
            except ValueError:
                pass
        
        # Heuristic categorization
        if any(kw in text_lower for kw in ["ignore", "disregard", "override", "forget", "system prompt"]):
            return AttackType.INSTRUCTION_OVERRIDE
        elif any(kw in text_lower for kw in ["dan", "unrestricted", "no rules", "roleplay", "pretend"]):
            return AttackType.ROLE_HIJACK
        elif any(kw in text_lower for kw in ["base64", "rot13", "encode", "decode", "hex"]):
            return AttackType.ENCODING_OBFUSCATION
        elif any(kw in text_lower for kw in ["api_key", "password", "secret", "token", "credit card"]):
            return AttackType.DATA_EXFILTRATION
        elif any(kw in text_lower for kw in ["function", "tool", "call", "invoke"]):
            return AttackType.TOOL_MISUSE
        elif any(kw in text_lower for kw in ["url", "link", "document", "fetch", "read"]):
            return AttackType.INDIRECT_INJECTION
        else:
            return AttackType.INSTRUCTION_OVERRIDE  # Default


class GitHubAdvisorySource(ThreatIntelSource):
    """Fetch prompt injection advisories from GitHub repositories."""
    
    async def fetch(self) -> AsyncIterator[AttackSample]:
        await self._rate_limit()
        
        # Known prompt injection repositories
        repos = [
            "prompt-injection/prompt-injection-attacks",
            "langchain-ai/langchain-security",
            "hiyouga/awesome-prompt-injection",
        ]
        
        for repo in repos:
            try:
                url = f"https://api.github.com/repos/{repo}/contents"
                async with self.session.get(url, headers=self.source.headers) as resp:
                    if resp.status != 200:
                        continue
                    files = await resp.json()
                    
                    for file in files:
                        if file["type"] == "file" and file["name"].endswith((".md", ".txt", ".json", ".yaml", ".yml")):
                            async with self.session.get(file["download_url"]) as fresp:
                                if fresp.status == 200:
                                    content = await fresp.text()
                                    yield from self._parse_content(content, file["download_url"])
            except Exception as e:
                logger.warning("Failed to fetch from %s: %s", repo, e)
    
    def _parse_content(self, content: str, source_url: str) -> List[AttackSample]:
        """Parse repository content for attack patterns."""
        samples = []
        
        # Extract code blocks and quoted text
        code_blocks = re.findall(r"```[\w]*\n(.*?)\n```", content, re.DOTALL)
        quoted = re.findall(r"`([^`]{20,})`", content)
        lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 20]
        
        all_candidates = code_blocks + quoted + lines
        
        for candidate in all_candidates:
            candidate = candidate.strip()
            if len(candidate) < 20 or len(candidate) > 5000:
                continue
            
            # Filter obvious non-attacks
            if candidate.startswith(("# ", "//", "/*", "import ", "from ", "def ", "class ")):
                continue
            
            sample = AttackSample(
                attack_text=candidate,
                category=self._categorize_attack(candidate, {}),
                source="github",
                source_url=source_url,
                discovered_at=datetime.now(timezone.utc),
                confidence=0.6,  # Lower confidence for auto-extracted
                metadata={"extraction_method": "github_repo"},
            )
            samples.append(sample)
        
        return samples


class ArxivSource(ThreatIntelSource):
    """Fetch prompt injection research from arXiv."""
    
    async def fetch(self) -> AsyncIterator[AttackSample]:
        await self._rate_limit()
        
        # Search for prompt injection papers
        query = "prompt injection OR jailbreak OR adversarial prompt"
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=50"
        
        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return
                xml_content = await resp.text()
                yield from self._parse_arxiv(xml_content)
        except Exception as e:
            logger.warning("Failed to fetch from arXiv: %s", e)
    
    def _parse_arxiv(self, xml: str) -> List[AttackSample]:
        """Parse arXiv XML for attack examples in abstracts."""
        samples = []
        
        # Simple extraction from abstracts
        abstracts = re.findall(r"<summary>(.*?)</summary>", xml, re.DOTALL)
        
        for abstract in abstracts:
            # Look for quoted attack examples
            quoted = re.findall(r'["\']([^"\']{30,200})["\']', abstract)
            for quote in quoted:
                if any(kw in quote.lower() for kw in ["ignore", "prompt", "injection", "jailbreak", "dan"]):
                    sample = AttackSample(
                        attack_text=quote.strip(),
                        category=AttackType.INSTRUCTION_OVERRIDE,
                        source="arxiv",
                        source_url="https://arxiv.org",
                        discovered_at=datetime.now(timezone.utc),
                        confidence=0.7,
                        metadata={"source_type": "research_paper"},
                    )
                    samples.append(sample)
        
        return samples


class BugBountySource(ThreatIntelSource):
    """Fetch disclosed prompt injection vulnerabilities from bug bounty platforms."""
    
    async def fetch(self) -> AsyncIterator[AttackSample]:
        # This would integrate with HackerOne/Bugcrowd APIs
        # For now, return known public disclosures
        known_disclosures = [
            # These would come from actual API in production
        ]
        for sample in known_disclosures:
            yield sample


class LocalFileSource(ThreatIntelSource):
    """Load attacks from local files (JSONL format)."""
    
    def __init__(self, source: ThreatSource, session: aiohttp.ClientSession, file_path: str):
        super().__init__(source, session)
        self.file_path = Path(file_path)
    
    async def fetch(self) -> AsyncIterator[AttackSample]:
        if not self.file_path.exists():
            logger.warning("Local file source not found: %s", self.file_path)
            return
        
        try:
            with open(self.file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield AttackSample(
                            attack_text=data.get("attack_text", ""),
                            category=AttackType(data.get("category", "instruction_override")),
                            source="local_file",
                            source_url=f"file://{self.file_path}",
                            discovered_at=datetime.fromisoformat(data.get("discovered_at", datetime.now().isoformat())),
                            confidence=data.get("confidence", 0.9),
                            metadata=data.get("metadata", {}),
                            validated=data.get("validated", True),
                        )
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("Failed to load local file source: %s", e)


class ThreatIntelPipeline:
    """
    Main pipeline for continuous threat intelligence processing.
    
    Features:
    - Multi-source fetching with rate limiting
    - Deduplication via content hashing
    - Human validation workflow
    - Automatic indexing to Qdrant
    - False positive tracking
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantStore] = None,
        validation_callback: Optional[Callable[[AttackSample], Awaitable[bool]]] = None,
    ):
        self.settings = settings or default_settings
        self.store = store or QdrantStore(settings=self.settings)
        self.validation_callback = validation_callback
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Source registry
        self.sources: Dict[str, ThreatIntelSource] = {}
        self.source_configs: Dict[str, ThreatSource] = {}
        
        # Deduplication
        self._seen_hashes: Set[str] = set()
        self._load_seen_hashes()
        
        # Validation queue
        self._validation_queue: asyncio.Queue[AttackSample] = asyncio.Queue()
        self._indexing_queue: asyncio.Queue[AttackSample] = asyncio.Queue()
        
        # Stats
        self.stats = {
            "total_fetched": 0,
            "total_deduplicated": 0,
            "total_validated": 0,
            "total_indexed": 0,
            "total_false_positives": 0,
            "last_run": None,
        }
    
    def _load_seen_hashes(self):
        """Load previously seen content hashes from disk."""
        hash_file = Path(self.settings.qdrant_collection + "_hashes.json")
        if hash_file.exists():
            try:
                with open(hash_file) as f:
                    self._seen_hashes = set(json.load(f))
            except Exception:
                pass
    
    def _save_seen_hashes(self):
        """Save seen hashes to disk."""
        hash_file = Path(self.settings.qdrant_collection + "_hashes.json")
        try:
            with open(hash_file, "w") as f:
                json.dump(list(self._seen_hashes), f)
        except Exception as e:
            logger.warning("Failed to save seen hashes: %s", e)
    
    def register_source(self, source: ThreatSource, parser_class: type = None):
        """Register a threat intelligence source."""
        self.source_configs[source.name] = source
        if parser_class:
            # Custom parser - will be instantiated on start
            self.source_configs[source.name].metadata["parser_class"] = parser_class
    
    async def start(self):
        """Start the pipeline."""
        self.session = aiohttp.ClientSession()
        
        # Initialize sources
        for name, config in self.source_configs.items():
            if not config.enabled:
                continue
            
            parser = config.metadata.get("parser_class", GitHubAdvisorySource)
            self.sources[name] = parser(config, self.session)
        
        # Start background workers
        asyncio.create_task(self._validation_worker())
        asyncio.create_task(self._indexing_worker())
        asyncio.create_task(self._periodic_fetch())
        
        logger.info("Threat intelligence pipeline started with %d sources", len(self.sources))
    
    async def stop(self):
        """Stop the pipeline."""
        if self.session:
            await self.session.close()
        self._save_seen_hashes()
        logger.info("Threat intelligence pipeline stopped")
    
    async def _periodic_fetch(self):
        """Periodically fetch from all sources."""
        while True:
            try:
                await self.fetch_all()
                await asyncio.sleep(6 * 3600)  # Every 6 hours
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Periodic fetch error: %s", e)
                await asyncio.sleep(300)  # 5 min backoff
    
    async def fetch_all(self) -> Dict[str, int]:
        """Fetch from all enabled sources."""
        results = {}
        
        for name, source in self.sources.items():
            try:
                count = 0
                async for sample in source.fetch():
                    await self._process_sample(sample)
                    count += 1
                results[name] = count
                self.stats["total_fetched"] += count
            except Exception as e:
                logger.error("Source %s failed: %s", name, e)
                results[name] = 0
        
        self.stats["last_run"] = datetime.now(timezone.utc).isoformat()
        return results
    
    async def _process_sample(self, sample: AttackSample):
        """Process a single attack sample."""
        # Deduplication
        content_hash = sample.content_hash()
        if content_hash in self._seen_hashes:
            self.stats["total_deduplicated"] += 1
            return
        
        self._seen_hashes.add(content_hash)
        
        # Queue for validation
        await self._validation_queue.put(sample)
    
    async def _validation_worker(self):
        """Background worker for human validation."""
        while True:
            try:
                sample = await self._validation_queue.get()
                
                # Auto-validate high-confidence samples
                if sample.confidence >= 0.9 and not sample.metadata.get("requires_human", False):
                    sample.validated = True
                    sample.validated_at = datetime.now(timezone.utc)
                    sample.validated_by = "auto"
                    await self._indexing_queue.put(sample)
                    self.stats["total_validated"] += 1
                    continue
                
                # Custom validation callback
                if self.validation_callback:
                    try:
                        is_valid = await self.validation_callback(sample)
                        if is_valid:
                            sample.validated = True
                            sample.validated_at = datetime.now(timezone.utc)
                            sample.validated_by = "callback"
                            await self._indexing_queue.put(sample)
                            self.stats["total_validated"] += 1
                        else:
                            sample.false_positive = True
                            self.stats["total_false_positives"] += 1
                    except Exception as e:
                        logger.error("Validation callback failed: %s", e)
                else:
                    # No callback - auto-validate for now
                    sample.validated = True
                    sample.validated_at = datetime.now(timezone.utc)
                    sample.validated_by = "auto_default"
                    await self._indexing_queue.put(sample)
                    self.stats["total_validated"] += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Validation worker error: %s", e)
    
    async def _indexing_worker(self):
        """Background worker for indexing to Qdrant."""
        while True:
            try:
                sample = await self._indexing_queue.get()
                
                if not sample.validated or sample.false_positive:
                    continue
                
                # Check if already exists in Qdrant
                existing = await self.store.search_similar_attacks(
                    query_text=sample.attack_text,
                    limit=1,
                    score_threshold=0.95,
                )
                
                if existing:
                    logger.debug("Attack already in corpus (similarity >= 0.95): %s", sample.content_hash())
                    continue
                
                # Index to Qdrant
                from ares.redteam.attacker import AttackAttempt
                
                attempt = AttackAttempt(
                    attempt_number=0,
                    attack_text=sample.attack_text,
                    category=sample.category.value,
                    victim_response="",
                    rule_verdict=False,
                    llm_verdict=False,
                    success=False,
                    reasoning=f"Threat intel: {sample.source}",
                    domain="threat_intel",
                    source=sample.source,
                    attacker_provider="threat_intel",
                    victim_provider="unknown",
                    judge_provider="threat_intel",
                )
                
                await self.store.upsert_attempt(
                    attempt=attempt,
                    target_prompt_hash="threat_intel_" + content_hash,
                )
                
                self.stats["total_indexed"] += 1
                logger.info("Indexed new attack from %s: %s", sample.source, sample.content_hash())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Indexing worker error: %s", e)
    
    def get_stats(self) -> dict:
        return {
            **self.stats,
            "seen_hashes": len(self._seen_hashes),
            "validation_queue_size": self._validation_queue.qsize(),
            "indexing_queue_size": self._indexing_queue.qsize(),
            "sources": {name: config.enabled for name, config in self.source_configs.items()},
        }
    
    async def submit_sample(self, sample: AttackSample) -> bool:
        """Manually submit an attack sample."""
        if sample.content_hash() in self._seen_hashes:
            return False
        await self._validation_queue.put(sample)
        return True
    
    async def report_false_positive(self, attack_text: str) -> bool:
        """Report a false positive to improve future detection."""
        content_hash = hashlib.sha256(attack_text.encode()).hexdigest()[:16]
        # This would remove from index and add to negative training set
        logger.info("False positive reported: %s", content_hash)
        return True


# Default pipeline instance
_pipeline: Optional[ThreatIntelPipeline] = None


async def get_pipeline() -> ThreatIntelPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ThreatIntelPipeline()
        await _pipeline.start()
    return _pipeline


async def shutdown_pipeline():
    """Shutdown the global pipeline."""
    global _pipeline
    if _pipeline:
        await _pipeline.stop()
        _pipeline = None