"""
ARES Per-Client Model Training Orchestrator & CLI.

Builds and registers isolated intent/threat classifiers and compression adapters
per organization (`models/{org_id}/`) while enforcing 8 GB VRAM safety by
unloading active Ollama models before training.

Usage:
    python -m ares.training.client_model_trainer --org=org_acme_fintech
    python -m ares.training.client_model_trainer --org=all
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ares.training.compression_adapter_trainer import CompressionAdapterTrainer
from ares.training.distilbert_intent_trainer import DistilBertIntentTrainer
from ares.training.model_registry import (
    DEFAULT_CLIENT_PROFILES,
    MODELS_ROOT_DIR,
    ModelRegistry,
    validate_org_id,
)

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CLIENT_ML_STATS_PATH = _BACKEND_DIR / "data" / "client_models_ml_stats.json"


def unload_ollama_models(ollama_url: str = "http://localhost:11434") -> None:
    """
    8 GB VRAM Guardrail: politely instruct Ollama to unload resident models
    (`keep_alive: 0`) before any local GPU/ML training begins.
    Fails open silently when Ollama is offline.
    """
    try:
        import urllib.request

        for model_name in ("qwen2.5:3b", "llama3.1:8b", "ares-optimizer:8b"):
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/generate",
                data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=1.0):
                    pass
            except Exception:
                pass
    except Exception:
        pass


def train_client_models(
    org_id: str,
    *,
    domain: Optional[str] = None,
    findings_path: Optional[Path] = None,
    models_root: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    version: str = "v1.0",
    use_transformers: bool = False,
    run_qlora: bool = False,
) -> Dict[str, Any]:
    """
    Execute the complete per-client training pipeline for `org_id` and register
    resulting artifacts in `model_registry.json`.
    """
    org_id = validate_org_id(org_id)
    unload_ollama_models()

    resolved_models_root = Path(models_root) if models_root else MODELS_ROOT_DIR
    registry = ModelRegistry(registry_path=registry_path)

    existing_cfg = registry.get_client_config(org_id) or {}
    effective_domain = domain or existing_cfg.get("domain")

    intent_trainer = DistilBertIntentTrainer(
        findings_path=findings_path,
        models_root=resolved_models_root,
    )
    intent_res = intent_trainer.train_for_org(
        org_id=org_id,
        domain=effective_domain,
        use_transformers=use_transformers,
    )

    comp_trainer = CompressionAdapterTrainer(models_root=resolved_models_root)
    comp_res = comp_trainer.train_for_org(
        org_id=org_id,
        domain=intent_res["domain"],
        run_qlora=run_qlora,
    )

    rel_intent_dir = f"models/{org_id}/intent_classifier/"
    rel_comp_dir = f"models/{org_id}/compression_adapter/"
    combined_metrics = {
        **intent_res["metrics"],
        "sft_pair_count": comp_res["sft_pair_count"],
        "avg_token_savings_pct": comp_res["avg_token_savings_pct"],
    }

    registered = registry.register_client_models(
        org_id=org_id,
        intent_model_path=rel_intent_dir,
        compression_model_path=rel_comp_dir,
        version=version,
        domain=intent_res["domain"],
        metrics=combined_metrics,
    )

    # Clear cached inference bundle if loaded in memory so new weights take effect immediately
    try:
        from ares.intent.per_client_classifier import clear_per_client_cache

        clear_per_client_cache(org_id)
    except Exception:
        pass

    return {
        "org_id": org_id,
        "domain": intent_res["domain"],
        "intent_model_dir": intent_res["model_dir"],
        "compression_adapter_dir": comp_res["adapter_dir"],
        "metrics": combined_metrics,
        "ml_stats": intent_res.get("ml_stats", {}),
        "registry_entry": registered,
    }


def export_consolidated_ml_stats(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Save consolidated per-client ML evaluation statistics to `backend/data/client_models_ml_stats.json`."""
    target = Path(output_path) if output_path else DEFAULT_CLIENT_ML_STATS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    existing_clients: Dict[str, Any] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded.get("clients"), dict):
                existing_clients = loaded["clients"]
        except Exception:
            existing_clients = {}

    for res in results:
        oid = res["org_id"]
        stats = res.get("ml_stats") or {}
        existing_clients[oid] = {
            "org_id": oid,
            "domain": res.get("domain", "general"),
            "summary_metrics": res.get("metrics", {}),
            "threat_ml_stats": stats.get("threat_ml_stats", {}),
            "intent_ml_stats": stats.get("intent_ml_stats", {}),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clients": existing_clients,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def format_ml_stats_summary(results: List[Dict[str, Any]]) -> str:
    """Render a clean terminal summary table with Accuracy, CV Accuracy, Precision, Recall, and F1."""
    lines = ["", "=" * 86, "  ARES PER-CLIENT ML EVALUATION STATISTICS (5-FOLD STRATIFIED CV + OUT-OF-FOLD)", "=" * 86]
    for res in results:
        oid = res["org_id"]
        domain = res.get("domain", "general")
        stats = res.get("ml_stats") or {}
        lines.append(f"\n[{oid}] (domain={domain})")
        for key, title in (("threat_ml_stats", "Threat Classifier"), ("intent_ml_stats", "Intent Classifier")):
            block = stats.get(key) or {}
            if not block:
                continue
            lines.append(
                f"  {title}: TrainAcc={block.get('overall_accuracy', 0)*100:.2f}% | "
                f"CV({block.get('cv_folds', 0)}-fold)={block.get('cv_mean_accuracy', 0)*100:.2f}% "
                f"(+-{block.get('cv_std_accuracy', 0)*100:.2f}%) | "
                f"Precision={block.get('macro_precision', 0)*100:.2f}% | "
                f"Recall={block.get('macro_recall', 0)*100:.2f}% | "
                f"F1={block.get('macro_f1', 0)*100:.2f}% (Weighted F1={block.get('weighted_f1', 0)*100:.2f}%)"
            )
            lines.append("    Category                   Precision    Recall        F1   Support")
            lines.append("    " + "-" * 66)
            for row in block.get("category_breakdown") or []:
                lines.append(
                    f"    {row['category']:<24} {row['precision']*100:>8.2f}% "
                    f"{row['recall']*100:>8.2f}% {row['f1_score']*100:>8.2f}% "
                    f"{row['support']:>9d}"
                )
    lines.append("=" * 86)
    return "\n".join(lines)


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train ARES per-client ML & compression models")
    parser.add_argument(
        "--org",
        required=True,
        help="Target organization ID (e.g. org_acme_fintech) or 'all' for all reference clients.",
    )
    parser.add_argument("--domain", default=None, help="Optional explicit domain override.")
    parser.add_argument("--version", default="v1.0", help="Model artifact version label.")
    parser.add_argument("--use-transformers", action="store_true", help="Run optional local DistilBERT fine-tuning.")
    parser.add_argument("--run-qlora", action="store_true", help="Run optional QLoRA adapter fine-tuning.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result before summary table.")
    args = parser.parse_args()

    org_ids: List[str]
    if args.org.lower() == "all":
        org_ids = list(DEFAULT_CLIENT_PROFILES.keys())
    else:
        org_ids = [args.org]

    results = []
    for oid in org_ids:
        res = train_client_models(
            oid,
            domain=args.domain,
            version=args.version,
            use_transformers=args.use_transformers,
            run_qlora=args.run_qlora,
        )
        results.append(res)

    stats_path = export_consolidated_ml_stats(results)
    if args.json:
        print(json.dumps({"trained_orgs": [r["org_id"] for r in results], "stats_report": str(stats_path), "results": results}, indent=2))
    print(format_ml_stats_summary(results))
    print(f"Saved consolidated ML statistics report to: {stats_path}")


if __name__ == "__main__":
    _cli()
