"""
Per-Client Compression Adapter Trainer.

Partitions domain-specific SFT optimization pairs from `sft_optimizer_train.jsonl`
and `SEED_EXEMPLARS`, derives client-specific lexical compression rules and
protected vocabulary, and builds `backend/models/{org_id}/compression_adapter/`
artifacts (`adapter_config.json`, `exemplars.json`, `training_dataset.jsonl`).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ares.intent.exemplar_store import SEED_EXEMPLARS
from ares.training.distilbert_intent_trainer import _DOMAIN_DATASET_KEYWORDS, _infer_domain
from ares.training.model_registry import MODELS_ROOT_DIR, validate_org_id

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SFT_TRAIN_PATH = _BACKEND_DIR / "ares" / "intent" / "data" / "sft_optimizer_train.jsonl"

_DOMAIN_LEXICAL_RULES: Dict[str, Dict[str, Any]] = {
    "fintech": {
        "protected_terms": [
            "SEC", "10-K", "10-Q", "EBITDA", "DCF", "SOX", "PCI-DSS", "GAAP", "Q1", "Q2", "Q3", "Q4",
        ],
        "lexical_replacements": {
            "at your earliest convenience": "",
            "due to the fact that": "because",
            "in order to": "to",
            "for the purpose of": "for",
            "at this point in time": "now",
            "with regards to": "re:",
            "quarterly earnings report": "quarterly earnings",
        },
        "target_compression_ratio": 0.40,
    },
    "software": {
        "protected_terms": [
            "kubectl", "docker", "terraform", "async", "def ", "class ", "POST ", "GET ", "JSON", "SQL",
        ],
        "lexical_replacements": {
            "at your earliest convenience": "",
            "due to the fact that": "because",
            "in order to": "to",
            "for the purpose of": "for",
            "at this point in time": "now",
            "without using any external third party libraries": "using only stdlib",
        },
        "target_compression_ratio": 0.45,
    },
    "healthcare": {
        "protected_terms": [
            "HIPAA", "PHI", "ICD-10", "mg", "mcg", "mL", "SpO2", "BP", "HR", "EGFR",
        ],
        "lexical_replacements": {
            "at your earliest convenience": "",
            "due to the fact that": "because",
            "in order to": "to",
            "for the purpose of": "for",
            "at this point in time": "now",
            "patient confidential health records": "PHI records",
        },
        "target_compression_ratio": 0.35,
    },
    "general": {
        "protected_terms": ["JSON", "XML", "CSV"],
        "lexical_replacements": {
            "at your earliest convenience": "",
            "due to the fact that": "because",
            "in order to": "to",
            "for the purpose of": "for",
            "at this point in time": "now",
        },
        "target_compression_ratio": 0.40,
    },
}


class CompressionAdapterTrainer:
    """Builds per-client compression adapter artifacts and optional PEFT LoRA fine-tuning."""

    def __init__(
        self,
        sft_train_path: Optional[Path] = None,
        models_root: Optional[Path] = None,
    ) -> None:
        self.sft_train_path = Path(sft_train_path) if sft_train_path else DEFAULT_SFT_TRAIN_PATH
        self.models_root = Path(models_root) if models_root else MODELS_ROOT_DIR

    def _partition_sft_records(self, domain: str, max_records: int = 250) -> Tuple[List[Dict[str, Any]], float]:
        """Select SFT optimization pairs prioritized for the client's domain."""
        keywords = _DOMAIN_DATASET_KEYWORDS.get(domain, _DOMAIN_DATASET_KEYWORDS["general"])
        domain_records: List[Dict[str, Any]] = []
        fallback_records: List[Dict[str, Any]] = []

        if self.sft_train_path.exists():
            with self.sft_train_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec_domain = str(rec.get("domain") or "general").strip().lower()
                    if any(kw in rec_domain for kw in keywords):
                        domain_records.append(rec)
                    elif len(fallback_records) < max_records:
                        fallback_records.append(rec)

        selected = (domain_records + fallback_records)[:max_records]
        if selected:
            avg_savings = sum(float(r.get("token_savings_pct", 0.35)) for r in selected) / len(selected)
        else:
            avg_savings = 0.40
        return selected, round(avg_savings, 4)

    def _build_domain_exemplars(self, domain: str, sft_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect structured few-shot compression exemplars tailored to `domain`."""
        exemplars: List[Dict[str, Any]] = []
        for ex in SEED_EXEMPLARS:
            if ex.domain.lower() == domain or domain == "general":
                exemplars.append(
                    {
                        "id": ex.id,
                        "intent": ex.intent.value,
                        "domain": ex.domain,
                        "original_prompt": ex.original_prompt,
                        "optimized_prompt": ex.optimized_prompt,
                        "token_savings_pct": ex.token_savings_pct,
                    }
                )

        # Pull up to 6 concise pairs directly from the partitioned SFT corpus
        for rec in sft_records:
            if len(exemplars) >= 10:
                break
            msgs = rec.get("messages") or []
            user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            asst_msg = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            if "ORIGINAL UNOPTIMIZED PROMPT:\n" in user_msg and asst_msg:
                orig_text = user_msg.split("ORIGINAL UNOPTIMIZED PROMPT:\n", 1)[1].split(
                    "\n\nREWRITTEN OPTIMIZED PROMPT:", 1
                )[0].strip()
                exemplars.append(
                    {
                        "id": rec.get("id", "sft_pair"),
                        "intent": rec.get("intent", "analytical_qa"),
                        "domain": rec.get("domain", domain),
                        "original_prompt": orig_text,
                        "optimized_prompt": asst_msg.strip(),
                        "token_savings_pct": float(rec.get("token_savings_pct", 0.40)),
                    }
                )
        return exemplars

    def train_for_org(
        self,
        org_id: str,
        domain: Optional[str] = None,
        run_qlora: bool = False,
    ) -> Dict[str, Any]:
        """Build per-client compression adapter artifacts under `models/{org_id}/compression_adapter/`."""
        org_id = validate_org_id(org_id)
        t0 = time.perf_counter()
        resolved_domain = _infer_domain(org_id, domain)
        resolved_root = self.models_root.resolve()
        output_dir = (resolved_root / org_id / "compression_adapter").resolve()
        if resolved_root not in output_dir.parents:
            raise ValueError(f"Adapter directory escaped models root for org_id '{org_id}'.")
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_records, avg_savings = self._partition_sft_records(resolved_domain)
        exemplars = self._build_domain_exemplars(resolved_domain, sft_records)
        rule_profile = _DOMAIN_LEXICAL_RULES.get(resolved_domain, _DOMAIN_LEXICAL_RULES["general"])

        dataset_path = output_dir / "training_dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as handle:
            for rec in sft_records:
                handle.write(json.dumps(rec) + "\n")

        exemplars_path = output_dir / "exemplars.json"
        exemplars_path.write_text(json.dumps(exemplars, indent=2), encoding="utf-8")

        qlora_executed = False
        if run_qlora and dataset_path.exists() and len(sft_records) > 0:
            try:
                from ares.training.train_lora import train_lora_adapter

                train_lora_adapter(
                    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
                    train_data_path=dataset_path,
                    val_data_path=None,
                    output_dir=output_dir,
                    num_epochs=1,
                    max_steps=20,
                    lora_r=8,
                    lora_alpha=16,
                    org_id=org_id,
                    models_root=self.models_root,
                )
                qlora_executed = True
            except Exception as exc:
                logger.warning("Optional QLoRA fine-tuning skipped for '%s': %s", org_id, exc)

        ares_metadata = {
            "org_id": org_id,
            "domain": resolved_domain,
            "protected_terms": rule_profile["protected_terms"],
            "lexical_replacements": rule_profile["lexical_replacements"],
            "target_compression_ratio": rule_profile["target_compression_ratio"],
            "observed_avg_token_savings": avg_savings,
            "sft_pair_count": len(sft_records),
            "exemplar_count": len(exemplars),
            "qlora_weights_trained": qlora_executed,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        # Always persist ARES lexical/telemetry metadata separately so PEFT adapter_config.json stays valid
        (output_dir / "ares_adapter_metadata.json").write_text(
            json.dumps(ares_metadata, indent=2),
            encoding="utf-8",
        )

        # Only write the fallback adapter_config.json when PEFT did not write real LoRA weights/config
        config_path = output_dir / "adapter_config.json"
        if not qlora_executed or not config_path.exists():
            adapter_config = {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                **ares_metadata,
            }
            config_path.write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")

        train_seconds = round(time.perf_counter() - t0, 3)
        logger.info(
            "Built compression adapter for '%s' (%s) with %d pairs in %.2fs",
            org_id,
            resolved_domain,
            len(sft_records),
            train_seconds,
        )
        return {
            "adapter_dir": str(output_dir),
            "domain": resolved_domain,
            "sft_pair_count": len(sft_records),
            "exemplar_count": len(exemplars),
            "avg_token_savings_pct": avg_savings,
            "train_seconds": train_seconds,
        }
