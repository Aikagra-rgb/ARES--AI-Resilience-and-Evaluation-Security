#!/usr/bin/env python3
"""
ARES Reproducible Security & Evaluation Benchmark (Offline Deterministic Suite).

Compares baselines (`unprotected_baseline`, `layer0_structural_only`, `ares_full_gateway`)
across both:
1. Reference Client Target Matrix (`tests.redteam.attack_generator.TargetAttackGenerator`)
2. Independent Unseen Holdout Corpus (`evaluation/holdout_eval_corpus.jsonl`)

Outputs strictly REDACTED evidence artifacts (no raw prompts, secrets, or PII):
- `evaluation/reports/security_benchmark_report.json`
- `evaluation/reports/SECURITY_EVALUATION_REPORT.md`
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from ares.api.schemas import EnforceActionEnum, EnforceRequest  # noqa: E402
from ares.config import Settings, settings  # noqa: E402
from ares.gateway.context import TenantContext  # noqa: E402
from ares.gateway.enforcer import RuntimeEnforcer  # noqa: E402
from ares.intent.taxonomy import PromptIntent  # noqa: E402
from ares.optimizer.pipeline import (  # noqa: E402
    UserPromptOptimizer,
    verify_directive_markers,
    verify_json_schema_preserved,
    verify_protected_terms_preserved,
)
from ares.security.error_sanitizer import sanitize_error_message  # noqa: E402
from ares.security.structural_detector import Decision as StructuralDecision  # noqa: E402
from ares.security.structural_detector import StructuralDetector  # noqa: E402
from ares.vectordb.embedder import GeminiEmbedder  # noqa: E402
from ares.vectordb.store import QdrantStore  # noqa: E402
from tests.redteam import TargetAttackGenerator  # noqa: E402
from tests.targets import app as targets_app  # noqa: E402

DEFAULT_HOLDOUT_PATH = _REPO_ROOT / "evaluation" / "holdout_eval_corpus.jsonl"
DEFAULT_OPTIMIZER_TASKS_PATH = _REPO_ROOT / "evaluation" / "optimizer_holdout_tasks.jsonl"
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "evaluation" / "corpus_manifest.json"
DEFAULT_REPORT_DIR = _REPO_ROOT / "evaluation" / "reports"

_ORG_DOMAIN_MAP = {
    "org_acme_fintech": "fintech",
    "org_dev_ops_agent": "software",
    "org_medshield_health": "healthcare",
    "default": "general",
}


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    """Compute Wilson score 95% confidence interval percentages."""
    if total <= 0:
        return {"low_pct": 0.0, "high_pct": 0.0}
    p = max(0.0, min(1.0, successes / total))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return {
        "low_pct": round(max(0.0, center - margin) * 100.0, 2),
        "high_pct": round(min(1.0, center + margin) * 100.0, 2),
    }


def sha256_normalized(path: Path) -> str:
    """Compute SHA-256 normalizing CRLF -> LF so checksums match identically across Windows & Linux CI."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    idx = (len(sorted_vals) - 1) * pct
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return round(sorted_vals[int(idx)], 2)
    weight = idx - lower
    return round(sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight, 2)


def _compute_binary_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for r in records if r["label"] == 1 and r["flagged"])
    fn = sum(1 for r in records if r["label"] == 1 and not r["flagged"])
    fp = sum(1 for r in records if r["label"] == 0 and r["flagged"])
    tn = sum(1 for r in records if r["label"] == 0 and not r["flagged"])

    attacks = tp + fn
    benign = fp + tn
    recall = tp / attacks if attacks else 0.0
    fnr = fn / attacks if attacks else 0.0
    fpr = fp / benign if benign else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    lats = sorted(float(r["latency_ms"]) for r in records)

    return {
        "total_samples": len(records),
        "attack_samples": attacks,
        "benign_samples": benign,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "catch_rate_pct": round(recall * 100.0, 2),
        "catch_rate_ci_95": _wilson_ci(tp, attacks),
        "fnr_pct": round(fnr * 100.0, 2),
        "fnr_ci_95": _wilson_ci(fn, attacks),
        "fpr_pct": round(fpr * 100.0, 2),
        "fpr_ci_95": _wilson_ci(fp, benign),
        "precision_pct": round(precision * 100.0, 2),
        "f1_pct": round(f1 * 100.0, 2),
        "latency_p50_ms": _percentile(lats, 0.50),
        "latency_p95_ms": _percentile(lats, 0.95),
        "latency_p99_ms": _percentile(lats, 0.99),
    }


def _group_breakdown(records: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get(key) or "unknown")].append(rec)

    breakdown: Dict[str, Dict[str, Any]] = {}
    for grp in sorted(grouped.keys()):
        items = grouped[grp]
        positives = [i for i in items if i["label"] == 1]
        negatives = [i for i in items if i["label"] == 0]
        tp = sum(1 for i in positives if i["flagged"])
        fn = len(positives) - tp
        fp = sum(1 for i in negatives if i["flagged"])
        tn = len(negatives) - fp
        breakdown[grp] = {
            "total": len(items),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "catch_rate_pct": round((tp / len(positives)) * 100.0, 2) if positives else None,
            "fnr_pct": round((fn / len(positives)) * 100.0, 2) if positives else None,
            "fpr_pct": round((fp / len(negatives)) * 100.0, 2) if negatives else None,
        }
    return breakdown


def _category_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return _group_breakdown(records, "category")


def _domain_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return _group_breakdown(records, "domain")


def _make_offline_enforcer() -> RuntimeEnforcer:
    mock_store = MagicMock(spec=QdrantStore)
    mock_store.search_similar_attacks = AsyncMock(return_value=[])
    mock_store.embedder = MagicMock()
    mock_store.embedder.embed_text = AsyncMock(return_value=[0.5] * 64)
    mock_store.embedder.embed_batch = AsyncMock(return_value=[[0.5] * 64] * 10)
    return RuntimeEnforcer(settings=settings, store=mock_store)


def _evaluate_holdout_suites(holdout_path: Path) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    detector = StructuralDetector()
    enforcer = _make_offline_enforcer()

    unprotected_records: List[Dict[str, Any]] = []
    layer0_records: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []
    redacted_misses: List[Dict[str, Any]] = []
    redacted_false_positives: List[Dict[str, Any]] = []

    async def _run_all() -> None:
        for row in rows:
            prompt = row["prompt"]
            digest = _prompt_digest(prompt)
            cat = row["category"]
            label = int(row["label"])
            org_id = row.get("org_id") or "default"

            # 1. Unprotected baseline (zero gateway defense)
            unprotected_records.append(
                {
                    "case_id": row["case_id"],
                    "category": cat,
                    "label": label,
                    "flagged": False,
                    "latency_ms": 0.0,
                    "prompt_sha256": digest,
                }
            )

            # 2. Layer-0 Structural Detector only
            t0 = time.perf_counter()
            det_res = detector.detect(prompt)
            l0_ms = (time.perf_counter() - t0) * 1000.0
            l0_flagged = det_res.decision in (StructuralDecision.BLOCK, StructuralDecision.CHALLENGE)
            layer0_records.append(
                {
                    "case_id": row["case_id"],
                    "category": cat,
                    "label": label,
                    "flagged": l0_flagged,
                    "latency_ms": round(l0_ms, 2),
                    "prompt_sha256": digest,
                }
            )

            # 3. ARES Full Gateway (Layer 0 + Heuristics + Layer 1.5 Per-Client ML + Outbound Policy)
            t1 = time.perf_counter()
            tctx = TenantContext(organization_id=org_id)
            req = EnforceRequest(prompt=prompt, application_name=f"Benchmark-{org_id}")
            decision = await enforcer.enforce(req, context=tctx)
            full_ms = (time.perf_counter() - t1) * 1000.0
            full_flagged = decision.action in (EnforceActionEnum.block, EnforceActionEnum.sanitize)
            domain_val = str(row.get("domain") or _ORG_DOMAIN_MAP.get(org_id, "general"))
            rec = {
                "case_id": row["case_id"],
                "category": cat,
                "domain": domain_val,
                "label": label,
                "org_id": org_id,
                "flagged": full_flagged,
                "action": decision.action.value,
                "severity": decision.severity.value,
                "matched_technique": sanitize_error_message(decision.matched_technique or "clean", max_len=64),
                "latency_ms": round(full_ms, 2),
                "prompt_sha256": digest,
                "prompt_length": len(prompt),
            }
            full_records.append(rec)

            if label == 1 and not full_flagged:
                redacted_misses.append(
                    {
                        "case_id": row["case_id"],
                        "category": cat,
                        "org_id": org_id,
                        "prompt_sha256": digest,
                        "action": decision.action.value,
                        "matched_technique": rec["matched_technique"],
                    }
                )
            elif label == 0 and full_flagged:
                redacted_false_positives.append(
                    {
                        "case_id": row["case_id"],
                        "category": cat,
                        "org_id": org_id,
                        "prompt_sha256": digest,
                        "action": decision.action.value,
                        "matched_technique": rec["matched_technique"],
                    }
                )

    asyncio.run(_run_all())

    return {
        "corpus_file": str(holdout_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "corpus_sha256": sha256_normalized(holdout_path),
        "baselines": {
            "unprotected_baseline": _compute_binary_metrics(unprotected_records),
            "layer0_structural_only": _compute_binary_metrics(layer0_records),
            "ares_full_gateway": _compute_binary_metrics(full_records),
        },
        "ares_category_breakdown": _category_breakdown(full_records),
        "ares_domain_breakdown": _domain_breakdown(full_records),
        "layer0_category_breakdown": _category_breakdown(layer0_records),
        "redacted_false_negatives": redacted_misses,
        "redacted_false_positives": redacted_false_positives,
    }


def _evaluate_optimizer_tasks(tasks_path: Path) -> Dict[str, Any]:
    """Evaluate prompt optimizer on immutable held-out tasks with separated rules vs LoRA reporting."""
    rows = [
        json.loads(ln)
        for ln in tasks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    offline_settings = Settings(
        DATABASE_URL=settings.database_url,
        ENVIRONMENT="test",
        ARES_USE_LORA_SERVER=False,
    )
    mock_emb = MagicMock(spec=GeminiEmbedder)
    mock_emb.embed_text = AsyncMock(return_value=[0.5] * 64)
    mock_attack_store = MagicMock()
    mock_attack_store.search_similar_attacks = AsyncMock(return_value=[])
    mock_ex_store = MagicMock()
    mock_ex_store.search_exemplars = AsyncMock(return_value=[])
    mock_ex_store.format_few_shot_block = MagicMock(return_value="")

    optimizer = UserPromptOptimizer(
        settings=offline_settings,
        embedder=mock_emb,
        attack_store=mock_attack_store,
        exemplar_store=mock_ex_store,
    )

    records: List[Dict[str, Any]] = []
    redacted_failures: List[Dict[str, Any]] = []

    async def _run_tasks() -> None:
        for row in rows:
            prompt = row["prompt"]
            digest = _prompt_digest(prompt)
            res = await optimizer.optimize_prompt(
                user_prompt=prompt,
                domain=row["domain"],
                org_id=row["org_id"],
                inference_mode="rules_adapter",
            )
            opt_text = res.optimized_prompt or ""
            intent_ok = res.intent.value == row["expected_intent"]

            terms = list(row.get("protected_terms") or [])
            terms_ok, missing_terms = verify_protected_terms_preserved(prompt, opt_text, terms)

            req_fields = [f.lower() for f in (row.get("required_schema_fields") or [])]
            schema_ok = True
            if row["expected_intent"] == PromptIntent.DATA_EXTRACTION_JSON.value:
                schema_ok = verify_json_schema_preserved(prompt, opt_text) and all(
                    f in opt_text.lower() for f in req_fields
                )

            directive_ok = True
            if row["expected_intent"] == PromptIntent.SYSTEM_DIRECTIVE.value:
                req_markers = [m.lower() for m in (row.get("required_directive_markers") or [])]
                directive_ok = verify_directive_markers(prompt, opt_text, PromptIntent.SYSTEM_DIRECTIVE) and all(
                    m in opt_text.lower() for m in req_markers
                )

            not_blocked = res.optimized_prompt is not None and res.engine_kind == "rules_adapter"
            structural_ok = bool(not_blocked and terms_ok and schema_ok and directive_ok)

            records.append(
                {
                    "task_id": row["task_id"],
                    "domain": row["domain"],
                    "expected_intent": row["expected_intent"],
                    "predicted_intent": res.intent.value,
                    "intent_ok": intent_ok,
                    "structural_ok": structural_ok,
                    "terms_ok": terms_ok,
                    "schema_ok": schema_ok,
                    "directive_ok": directive_ok,
                    "token_savings_pct": round(res.token_savings_pct * 100.0, 2),
                    "latency_ms": float(res.latency_ms),
                    "engine_kind": res.engine_kind,
                    "prompt_sha256": digest,
                }
            )
            if not (intent_ok and structural_ok):
                redacted_failures.append(
                    {
                        "task_id": row["task_id"],
                        "domain": row["domain"],
                        "expected_intent": row["expected_intent"],
                        "predicted_intent": res.intent.value,
                        "structural_ok": structural_ok,
                        "missing_terms": missing_terms,
                        "prompt_sha256": digest,
                    }
                )

    asyncio.run(_run_tasks())

    total = len(records)
    intent_hits = sum(1 for r in records if r["intent_ok"])
    struct_hits = sum(1 for r in records if r["structural_ok"])
    terms_hits = sum(1 for r in records if r["terms_ok"])
    pos_savings = sum(1 for r in records if r["token_savings_pct"] > 0.0)
    mean_savings = round(sum(r["token_savings_pct"] for r in records) / total, 2) if total else 0.0
    lats = sorted(r["latency_ms"] for r in records)

    per_domain: Dict[str, Dict[str, Any]] = {}
    for dom in sorted({r["domain"] for r in records}):
        d_recs = [r for r in records if r["domain"] == dom]
        per_domain[dom] = {
            "total": len(d_recs),
            "intent_accuracy_pct": round(sum(1 for r in d_recs if r["intent_ok"]) / len(d_recs) * 100.0, 2),
            "structural_pass_pct": round(sum(1 for r in d_recs if r["structural_ok"]) / len(d_recs) * 100.0, 2),
            "mean_token_savings_pct": round(sum(r["token_savings_pct"] for r in d_recs) / len(d_recs), 2),
        }

    per_intent: Dict[str, Dict[str, Any]] = {}
    for ic in sorted({r["expected_intent"] for r in records}):
        i_recs = [r for r in records if r["expected_intent"] == ic]
        per_intent[ic] = {
            "total": len(i_recs),
            "intent_accuracy_pct": round(sum(1 for r in i_recs if r["intent_ok"]) / len(i_recs) * 100.0, 2),
            "structural_pass_pct": round(sum(1 for r in i_recs if r["structural_ok"]) / len(i_recs) * 100.0, 2),
            "mean_token_savings_pct": round(sum(r["token_savings_pct"] for r in i_recs) / len(i_recs), 2),
        }

    return {
        "corpus_file": str(tasks_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "corpus_sha256": sha256_normalized(tasks_path),
        "rules_adapter_evaluation": {
            "engine_kind": "rules_adapter",
            "total_tasks": total,
            "intent_accuracy_pct": round(intent_hits / total * 100.0, 2) if total else 0.0,
            "intent_accuracy_ci_95": _wilson_ci(intent_hits, total),
            "structural_pass_rate_pct": round(struct_hits / total * 100.0, 2) if total else 0.0,
            "structural_pass_ci_95": _wilson_ci(struct_hits, total),
            "protected_terms_pass_rate_pct": round(terms_hits / total * 100.0, 2) if total else 0.0,
            "mean_token_savings_pct": mean_savings,
            "positive_compression_rate_pct": round(pos_savings / total * 100.0, 2) if total else 0.0,
            "latency_p50_ms": _percentile(lats, 0.50),
            "latency_p99_ms": _percentile(lats, 0.99),
            "per_domain_breakdown": per_domain,
            "per_intent_breakdown": per_intent,
            "redacted_failures": redacted_failures,
            "semantic_fidelity_note": (
                "Offline CI uses deterministic structural checks without external semantic embeddings; "
                "structural invariants are measured directly rather than synthetic 0.90 scores."
            ),
        },
        "lora_serving": {
            "status": "not_measured_offline",
            "client_lora_adapter_tested": False,
            "global_lora_tested": False,
            "note": (
                "Native PEFT LoRA server (:11435) offline in CI; rules_adapter metrics are never "
                "reported as LoRA model performance."
            ),
        },
    }


def _evaluate_target_matrix() -> Dict[str, Any]:
    attacks = TargetAttackGenerator.generate_attack_cases()
    benigns = TargetAttackGenerator.generate_benign_cases()

    unprotected_records: List[Dict[str, Any]] = []
    ares_records: List[Dict[str, Any]] = []

    with TestClient(targets_app) as client:
        for case in attacks:
            digest = _prompt_digest(case.query)
            base_resp = client.post(
                case.endpoint,
                json={
                    "query": case.query,
                    "gateway_enabled": False,
                    "use_mock_llm": True,
                    "include_poisoned_doc": case.include_poisoned_doc,
                },
            ).json()
            breached = bool(base_resp.get("canary_leaked") or base_resp.get("tool_hijacked"))
            unprotected_records.append(
                {
                    "case_id": case.case_id,
                    "category": case.attack_category,
                    "label": 1,
                    "flagged": not breached,
                    "latency_ms": 0.0,
                    "prompt_sha256": digest,
                }
            )

            t0 = time.perf_counter()
            prot_resp = client.post(
                case.endpoint,
                json={
                    "query": case.query,
                    "gateway_enabled": True,
                    "use_mock_llm": True,
                    "include_poisoned_doc": case.include_poisoned_doc,
                },
            ).json()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ares_records.append(
                {
                    "case_id": case.case_id,
                    "category": case.attack_category,
                    "label": 1,
                    "flagged": bool(prot_resp.get("blocked") is True or prot_resp.get("action") in ("BLOCK", "SANITIZE")),
                    "latency_ms": round(elapsed_ms, 2),
                    "prompt_sha256": digest,
                }
            )

        for case in benigns:
            digest = _prompt_digest(case.query)
            unprotected_records.append(
                {
                    "case_id": case.case_id,
                    "category": "benign",
                    "label": 0,
                    "flagged": False,
                    "latency_ms": 0.0,
                    "prompt_sha256": digest,
                }
            )

            t0 = time.perf_counter()
            ben_resp = client.post(
                case.endpoint,
                json={
                    "query": case.query,
                    "gateway_enabled": True,
                    "use_mock_llm": True,
                },
            ).json()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            ares_records.append(
                {
                    "case_id": case.case_id,
                    "category": "benign",
                    "label": 0,
                    "flagged": bool(ben_resp.get("blocked") is True or ben_resp.get("action") != "ALLOW"),
                    "latency_ms": round(elapsed_ms, 2),
                    "prompt_sha256": digest,
                }
            )

    return {
        "baselines": {
            "unprotected_targets": _compute_binary_metrics(unprotected_records),
            "ares_protected_targets": _compute_binary_metrics(ares_records),
        },
        "ares_category_breakdown": _category_breakdown(ares_records),
    }


def _render_markdown_report(report: Dict[str, Any]) -> str:
    hold = report["holdout_evaluation"]
    tgt = report["target_matrix_evaluation"]
    three = report["three_score_evaluation"]
    opt_eval = three["prompt_optimizer_quality"]["rules_adapter_evaluation"]
    lora_eval = three["prompt_optimizer_quality"]["lora_serving"]
    frozen = report["frozen_configuration"]
    h_base = hold["baselines"]
    t_base = tgt["baselines"]
    gw_full = h_base["ares_full_gateway"]

    lines = [
        "# ARES Reproducible Three-Score Evaluation Report",
        "",
        f"- **Generated At (UTC)**: `{report['generated_at']}`",
        "- **Mode**: Offline deterministic evaluation (no raw prompts, secrets, or PII stored; prompts represented strictly via 16-char SHA-256 hashes)",
        f"- **Runtime Stack**: Python `{frozen['python_version']}` · scikit-learn `{frozen['sklearn_version']}` · numpy `{frozen['numpy_version']}`",
        f"- **Frozen Thresholds**: `block={frozen['gateway_block_threshold']}`, `sanitize={frozen['gateway_sanitize_threshold']}`, `model_registry_clients={len(frozen['client_models'])}`",
        f"- **Gateway Holdout SHA-256**: `{hold['corpus_sha256']}`",
        f"- **Optimizer Holdout SHA-256**: `{three['prompt_optimizer_quality']['corpus_sha256']}`",
        "",
        "## Score 1: Gateway Security — Unseen Holdout Corpus (N="
        + str(gw_full["total_samples"])
        + ")",
        "",
        "| Configuration | Catch Rate (%) [95% CI] | FNR (%) | FPR (%) [95% CI] | Precision (%) | F1 (%) | p50 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("unprotected_baseline", "layer0_structural_only", "ares_full_gateway"):
        m = h_base[name]
        c_ci = m["catch_rate_ci_95"]
        f_ci = m["fpr_ci_95"]
        lines.append(
            f"| `{name}` | {m['catch_rate_pct']:.2f} [{c_ci['low_pct']:.1f}–{c_ci['high_pct']:.1f}] | "
            f"{m['fnr_pct']:.2f} | {m['fpr_pct']:.2f} [{f_ci['low_pct']:.1f}–{f_ci['high_pct']:.1f}] | "
            f"{m['precision_pct']:.2f} | {m['f1_pct']:.2f} | {m['latency_p50_ms']:.2f} | {m['latency_p99_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Gateway Per-Domain & Per-Category FP/FN Breakdown (`ares_full_gateway`)",
            "",
            "| Domain | Total | TP | FN | FP | TN | Catch Rate (%) | FNR (%) | FPR (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dom, stats in hold["ares_domain_breakdown"].items():
        cr = f"{stats['catch_rate_pct']:.2f}" if stats["catch_rate_pct"] is not None else "—"
        fnr = f"{stats['fnr_pct']:.2f}" if stats["fnr_pct"] is not None else "—"
        fpr = f"{stats['fpr_pct']:.2f}" if stats["fpr_pct"] is not None else "—"
        lines.append(
            f"| `{dom}` | {stats['total']} | {stats['tp']} | {stats['fn']} | {stats['fp']} | {stats['tn']} | {cr} | {fnr} | {fpr} |"
        )

    lines.extend(
        [
            "",
            "| Category | Total | TP | FN | FP | TN | Catch Rate (%) | FNR (%) | FPR (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cat, stats in hold["ares_category_breakdown"].items():
        cr = f"{stats['catch_rate_pct']:.2f}" if stats["catch_rate_pct"] is not None else "—"
        fnr = f"{stats['fnr_pct']:.2f}" if stats["fnr_pct"] is not None else "—"
        fpr = f"{stats['fpr_pct']:.2f}" if stats["fpr_pct"] is not None else "—"
        lines.append(
            f"| `{cat}` | {stats['total']} | {stats['tp']} | {stats['fn']} | {stats['fp']} | {stats['tn']} | {cr} | {fnr} | {fpr} |"
        )

    rt_sim = three["redteam_campaign_realism"]["reference_simulator_regression"]
    rt_live = three["redteam_campaign_realism"]["external_live_target"]
    lines.extend(
        [
            "",
            "## Score 2: Red-Team Campaign Realism (Separated from Simulator Regression)",
            "",
            f"- **External Live Target Realism (`{rt_live['target_mode']}`)**: **{rt_live['status']}** — {rt_live['note']}",
            f"- **Reference Simulator Regression (`{rt_sim['target_mode']}`, N={t_base['ares_protected_targets']['total_samples']})**: deterministic regression signal only (`is_security_proof=False`).",
            "",
            "| Configuration | Target Mode | Attacks Caught (%) | FNR (%) | FPR (%) | p50 (ms) | p99 (ms) |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| `unprotected_targets` | `reference-simulator` | {t_base['unprotected_targets']['catch_rate_pct']:.2f} | "
            f"{t_base['unprotected_targets']['fnr_pct']:.2f} | {t_base['unprotected_targets']['fpr_pct']:.2f} | — | — |",
            f"| `ares_protected_targets` | `reference-simulator` | {t_base['ares_protected_targets']['catch_rate_pct']:.2f} | "
            f"{t_base['ares_protected_targets']['fnr_pct']:.2f} | {t_base['ares_protected_targets']['fpr_pct']:.2f} | "
            f"{t_base['ares_protected_targets']['latency_p50_ms']:.2f} | {t_base['ares_protected_targets']['latency_p99_ms']:.2f} |",
            "",
            "## Score 3: Prompt Optimizer Quality — Customer Holdout Tasks (N="
            + str(opt_eval["total_tasks"])
            + ")",
            "",
            f"- **Engine Evaluated**: `{opt_eval['engine_kind']}` (lexical Stage-1 rules & per-client protected vocabulary)",
            f"- **Intent Routing Accuracy**: **{opt_eval['intent_accuracy_pct']:.2f}%** [95% CI: {opt_eval['intent_accuracy_ci_95']['low_pct']:.1f}–{opt_eval['intent_accuracy_ci_95']['high_pct']:.1f}%]",
            f"- **Structural Correctness Pass Rate**: **{opt_eval['structural_pass_rate_pct']:.2f}%** [95% CI: {opt_eval['structural_pass_ci_95']['low_pct']:.1f}–{opt_eval['structural_pass_ci_95']['high_pct']:.1f}%]",
            f"- **Protected Terms Preserved**: **{opt_eval['protected_terms_pass_rate_pct']:.2f}%** · **Mean Token Savings**: **{opt_eval['mean_token_savings_pct']:.2f}%** · **Latency (p50/p99)**: `{opt_eval['latency_p50_ms']:.2f} / {opt_eval['latency_p99_ms']:.2f} ms`",
            f"- **LoRA Serving (`client_lora_adapter` / `global_lora`)**: **{lora_eval['status']}** — {lora_eval['note']}",
            "",
            "| Domain | Tasks | Intent Accuracy (%) | Structural Pass (%) | Mean Savings (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dom, d_stats in opt_eval["per_domain_breakdown"].items():
        lines.append(
            f"| `{dom}` | {d_stats['total']} | {d_stats['intent_accuracy_pct']:.2f} | "
            f"{d_stats['structural_pass_pct']:.2f} | {d_stats['mean_token_savings_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Known Limitations & Scope Boundaries",
            "",
            "- Gateway holdout evaluation uses mocked Qdrant similarity (`[]`) so CI results stay deterministic offline without external API keys.",
            "- Reference-simulator red-team targets verify deterministic regression protection only and are never combined with live target catch rates.",
            "- Optimizer evaluation measures `rules_adapter` deterministic compaction separately from GPU LoRA serving (`not_measured_offline` when port 11435 is offline).",
            "- Reports intentionally exclude raw prompt strings to protect confidentiality; correlate `case_id` / `task_id` + `prompt_sha256` against operator-controlled corpora.",
            "",
        ]
    )
    return "\n".join(lines)


def build_and_write_reports(
    *,
    holdout_path: Path = DEFAULT_HOLDOUT_PATH,
    optimizer_tasks_path: Path = DEFAULT_OPTIMIZER_TASKS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_REPORT_DIR,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_file = _BACKEND_DIR / "ares" / "training" / "model_registry.json"
    registry_data = json.loads(reg_file.read_text(encoding="utf-8")) if reg_file.exists() else {}

    manifest_data = {}
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    holdout_res = _evaluate_holdout_suites(holdout_path)
    target_res = _evaluate_target_matrix()
    opt_res = _evaluate_optimizer_tasks(optimizer_tasks_path)

    client_models_snapshot = {
        oid: {
            "version": cfg.get("version"),
            "domain": cfg.get("domain"),
            "metrics": cfg.get("metrics", {}),
        }
        for oid, cfg in (registry_data.get("clients") or {}).items()
    }

    three_score_evaluation = {
        "gateway_security": {
            "corpus_file": holdout_res["corpus_file"],
            "corpus_sha256": holdout_res["corpus_sha256"],
            "baselines": holdout_res["baselines"],
            "per_category_fp_fn": holdout_res["ares_category_breakdown"],
            "per_domain_fp_fn": holdout_res["ares_domain_breakdown"],
            "redacted_false_negatives": holdout_res["redacted_false_negatives"],
            "redacted_false_positives": holdout_res["redacted_false_positives"],
        },
        "redteam_campaign_realism": {
            "external_live_target": {
                "status": "not_measured",
                "target_mode": "external-live-target",
                "note": "No live authorized external client target URL was supplied during offline deterministic CI benchmark.",
            },
            "reference_simulator_regression": {
                "target_mode": "reference-simulator",
                "is_security_proof": False,
                "metrics": target_res["baselines"]["ares_protected_targets"],
                "unprotected_baseline": target_res["baselines"]["unprotected_targets"],
                "note": "Deterministic reference target regression checks only; never conflated with external live target realism.",
            },
        },
        "prompt_optimizer_quality": opt_res,
    }

    report: Dict[str, Any] = {
        "schema_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest_version": manifest_data.get("corpus_version", "1.0.0"),
        "corpus_license": manifest_data.get("license", "MIT"),
        "redacted": True,
        "frozen_configuration": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "gateway_block_threshold": settings.gateway_block_threshold,
            "gateway_sanitize_threshold": settings.gateway_sanitize_threshold,
            "optimizer_lora_adapter": (registry_data.get("adapter_path") or "default"),
            "client_models": client_models_snapshot,
            "source_checksums": {
                "holdout_eval_corpus.jsonl": holdout_res["corpus_sha256"],
                "optimizer_holdout_tasks.jsonl": opt_res["corpus_sha256"],
                "attack_generator.py": sha256_normalized(
                    _BACKEND_DIR / "tests" / "redteam" / "attack_generator.py"
                ),
                "payloads.py": sha256_normalized(_BACKEND_DIR / "ares" / "redteam" / "payloads.py"),
            },
        },
        "three_score_evaluation": three_score_evaluation,
        "holdout_evaluation": holdout_res,
        "target_matrix_evaluation": target_res,
    }

    json_out = output_dir / "security_benchmark_report.json"
    md_out = output_dir / "SECURITY_EVALUATION_REPORT.md"
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_out.write_text(_render_markdown_report(report), encoding="utf-8")
    return {"report": report, "json_path": json_out, "md_path": md_out}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ARES reproducible three-score offline evaluation.")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT_PATH)
    parser.add_argument("--optimizer-tasks", type=Path, default=DEFAULT_OPTIMIZER_TASKS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    out = build_and_write_reports(
        holdout_path=args.holdout,
        optimizer_tasks_path=args.optimizer_tasks,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    three = out["report"]["three_score_evaluation"]
    h = three["gateway_security"]["baselines"]["ares_full_gateway"]
    rt = three["redteam_campaign_realism"]["reference_simulator_regression"]["metrics"]
    opt = three["prompt_optimizer_quality"]["rules_adapter_evaluation"]
    print(
        f"Score 1 (Gateway Security): CatchRate={h['catch_rate_pct']}% CI={h['catch_rate_ci_95']} "
        f"FNR={h['fnr_pct']}% FPR={h['fpr_pct']}%"
    )
    print(
        f"Score 2 (Red-Team Sim Regression): CatchRate={rt['catch_rate_pct']}% "
        f"LiveTarget={three['redteam_campaign_realism']['external_live_target']['status']}"
    )
    print(
        f"Score 3 (Optimizer rules_adapter): IntentAcc={opt['intent_accuracy_pct']}% "
        f"StructPass={opt['structural_pass_rate_pct']}% Savings={opt['mean_token_savings_pct']}% "
        f"LoRA={three['prompt_optimizer_quality']['lora_serving']['status']}"
    )
    print(f"Wrote redacted reports: {out['json_path']} and {out['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
