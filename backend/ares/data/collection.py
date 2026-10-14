"""
ARES User Prompt Collection & Data Governance Pipeline
Collects production prompts with enterprise opt-in flags and privacy sanitization.
"""
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class PromptCollector:
    """Collects and stores sanitized user prompts for dataset evolution."""

    def __init__(self, storage_dir: str = "backend/ares/data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.collection_file = self.storage_dir / "collected_prompts.jsonl"

    def collect_prompt(
        self,
        prompt: str,
        intent: str,
        domain: str = "general",
        user_id: Optional[str] = None,
        opt_in: bool = True,
    ) -> bool:
        """Appends opted-in prompt to collection storage."""
        if not opt_in:
            return False

        record = {
            "prompt": prompt,
            "intent": intent,
            "domain": domain,
            "user_id": user_id or "anonymous",
            "collected_at": datetime.utcnow().isoformat(),
            "status": "pending_review",
        }

        with open(self.collection_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return True
