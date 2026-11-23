"""
Closed-Loop Feedback-to-Training Exporter.

Merges:
  1. Approved human-moderated attack samples from `ModerationQueueItem` (and/or in-memory queue)
  2. Completed Arena `TestRun.findings` from SQLAlchemy
into `backend/data/redteam_findings.json` with idempotent deduplication, and optionally
triggers `train_client_models` to refresh per-client classifiers and compression adapters.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ares.moderation.models import ModerationItem, ModerationStatus
from ares.training.model_registry import DEFAULT_CLIENT_PROFILES, ModelRegistry

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FINDINGS_PATH = _BACKEND_DIR / "data" / "redteam_findings.json"

_DOMAIN_DEFAULT_ORG = {
    "fintech": "org_acme_fintech",
    "finance": "org_acme_fintech",
    "software": "org_dev_ops_agent",
    "devops": "org_dev_ops_agent",
    "healthcare": "org_medshield_health",
    "medical": "org_medshield_health",
}

_SEVERITY_TO_SCORE = {
    "critical": 95,
    "high": 85,
    "medium": 65,
    "low": 35,
    "safe": 0,
}


def _resolve_item_org(item: ModerationItem) -> str:
    meta = item.metadata or {}
    explicit = meta.get("org_id") or meta.get("organization_id")
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    domain_key = (item.domain or "general").strip().lower()
    return _DOMAIN_DEFAULT_ORG.get(domain_key, "default")


def _load_approved_moderation_from_db(
    session_factory: Callable[[], Any],
) -> List[ModerationItem]:
    from sqlalchemy import select
    from app.models import ModerationQueueItem
    from ares.moderation.queue import _orm_to_item

    with session_factory() as db:
        rows = db.scalars(
            select(ModerationQueueItem)
            .where(ModerationQueueItem.status == ModerationStatus.APPROVED.value)
            .order_by(ModerationQueueItem.submitted_at.asc())
        ).all()
        return [_orm_to_item(r) for r in rows]


def _load_completed_arena_findings_from_db(
    session_factory: Callable[[], Any],
    org_id_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.models import Organization, TestRun, TestRunStatus

    records: List[Dict[str, Any]] = []
    with session_factory() as db:
        org_names: Dict[str, str] = {
            str(o.id): str(o.name) for o in db.scalars(select(Organization)).all()
        }
        runs = db.scalars(
            select(TestRun)
            .options(selectinload(TestRun.findings), selectinload(TestRun.evidence))
            .where(TestRun.status.in_([TestRunStatus.completed, TestRunStatus.blocked]))
            .order_by(TestRun.created_at.asc())
        ).all()

        for run in runs:
            cfg = run.configuration or {}
            # Skip synthetic gateway audit rows unless explicitly Arena/evaluation runs
            if cfg.get("source") == "runtime_gateway":
                continue

            resolved_org = str(
                cfg.get("org_id")
                or cfg.get("organization_id")
                or org_names.get(str(run.organization_id))
                or run.organization_id
                or "default"
            )
            if org_id_filter and resolved_org != org_id_filter:
                continue

            evidence_by_cat: Dict[str, str] = {}
            for ev in run.evidence or []:
                if ev.summary and ev.category not in evidence_by_cat:
                    evidence_by_cat[ev.category] = str(ev.summary).strip()

            for finding in run.findings or []:
                prompt_text = (finding.safe_summary or "").strip() or evidence_by_cat.get(
                    finding.category, ""
                )
                if not prompt_text:
                    continue

                sev_val = (
                    finding.severity.value.lower()
                    if hasattr(finding.severity, "value")
                    else str(finding.severity).lower()
                )
                is_attack = bool(finding.attack_succeeded or int(finding.risk_score or 0) >= 60)
                if run.status == TestRunStatus.blocked:
                    decision_str = "BLOCK"
                    blocked_flag = True
                elif is_attack:
                    decision_str = "ALLOW" if finding.attack_succeeded else "BLOCK"
                    blocked_flag = not bool(finding.attack_succeeded)
                else:
                    decision_str = "ALLOW"
                    blocked_flag = False
                records.append(
                    {
                        "case_id": f"arena_{str(finding.id)[:12]}",
                        "organization_id": resolved_org,
                        "target_name": str(cfg.get("target_application") or "ARES Arena"),
                        "attack_category": str(finding.category or "prompt_injection"),
                        "prompt": prompt_text,
                        "label": 1 if is_attack else 0,
                        "decision": decision_str,
                        "blocked": blocked_flag,
                        "matched_technique": str(finding.category or "arena_finding"),
                        "severity": sev_val,
                        "risk_score": int(finding.risk_score or 0),
                        "latency_ms": float(run.duration_milliseconds or 0.0),
                        "source": "arena_test_run",
                        "test_run_id": str(run.id),
                        "finding_id": str(finding.id),
                    }
                )
    return records


def export_feedback_to_training(
    *,
    findings_path: Optional[Path] = None,
    org_id: Optional[str] = None,
    moderation_items: Optional[Iterable[ModerationItem]] = None,
    session_factory: Optional[Callable[[], Any]] = None,
    use_db_session: bool = True,
    include_moderation: bool = True,
    include_arena_findings: bool = True,
    retrain: bool = False,
    retrain_orgs: Optional[List[str]] = None,
    models_root: Optional[Path] = None,
    registry_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Merge approved moderation items and completed Arena TestRun.findings into
    `redteam_findings.json` and optionally trigger `train_client_models`.
    """
    target_path = Path(findings_path) if findings_path else DEFAULT_FINDINGS_PATH
    factory = session_factory
    if factory is None and use_db_session:
        try:
            from app.database import SessionLocal

            factory = SessionLocal
        except Exception:
            factory = None

    # 1. Load existing findings report cleanly
    if target_path.exists():
        try:
            raw_loaded = json.loads(target_path.read_text(encoding="utf-8"))
            if isinstance(raw_loaded, dict):
                report = raw_loaded
            elif isinstance(raw_loaded, list):
                report = {"findings": raw_loaded}
            else:
                report = {}
        except Exception:
            report = {}
    else:
        report = {}

    existing_findings: List[Dict[str, Any]] = list(report.get("findings") or [])
    existing_count = len(existing_findings)

    seen_case_ids: Set[str] = {
        str(item.get("case_id")) for item in existing_findings if item.get("case_id")
    }
    seen_prompt_keys: Set[Tuple[str, str]] = {
        (
            str(item.get("organization_id") or "default").strip(),
            str(item.get("prompt") or "").strip().lower(),
        )
        for item in existing_findings
        if str(item.get("prompt") or "").strip()
    }

    moderation_added = 0
    arena_added = 0
    duplicates_skipped = 0
    affected_orgs: Set[str] = set()

    # 2. Collect approved moderation items (from explicit list + DB)
    if include_moderation:
        combined_mod_items: Dict[str, ModerationItem] = {}
        if moderation_items is not None:
            for item in moderation_items:
                if item.status == ModerationStatus.APPROVED:
                    combined_mod_items[item.id] = item
        if factory is not None:
            try:
                for db_item in _load_approved_moderation_from_db(factory):
                    combined_mod_items.setdefault(db_item.id, db_item)
            except Exception as exc:
                logger.warning("Could not query moderation_queue_items table: %s", exc)

        for item in combined_mod_items.values():
            item_org = _resolve_item_org(item)
            if org_id and item_org != org_id:
                continue
            prompt_text = (item.attack_text or "").strip()
            if not prompt_text:
                continue
            case_id = f"mod_{item.id[:12]}"
            prompt_key = (item_org, prompt_text.lower())
            if case_id in seen_case_ids or prompt_key in seen_prompt_keys:
                duplicates_skipped += 1
                continue

            sev_lower = str(item.severity or "high").lower()
            existing_findings.append(
                {
                    "case_id": case_id,
                    "organization_id": item_org,
                    "target_name": str(
                        (item.metadata or {}).get("target_name") or item.source or "moderation_queue"
                    ),
                    "attack_category": str(item.category or "prompt_injection"),
                    "prompt": prompt_text,
                    "label": 1,
                    "decision": "BLOCK",
                    "blocked": True,
                    "matched_technique": str(item.category or "moderated_attack"),
                    "severity": sev_lower,
                    "risk_score": _SEVERITY_TO_SCORE.get(sev_lower, 85),
                    "latency_ms": 0.0,
                    "source": "moderation_queue",
                    "moderation_item_id": item.id,
                }
            )
            seen_case_ids.add(case_id)
            seen_prompt_keys.add(prompt_key)
            moderation_added += 1
            if item_org and item_org != "default":
                affected_orgs.add(item_org)

    # 3. Collect completed Arena TestRun.findings
    if include_arena_findings and factory is not None:
        try:
            arena_rows = _load_completed_arena_findings_from_db(factory, org_id_filter=org_id)
        except Exception as exc:
            logger.warning("Could not query Arena TestRun findings: %s", exc)
            arena_rows = []

        for row in arena_rows:
            case_id = str(row["case_id"])
            item_org = str(row["organization_id"])
            prompt_key = (item_org, str(row["prompt"]).strip().lower())
            if case_id in seen_case_ids or prompt_key in seen_prompt_keys:
                duplicates_skipped += 1
                continue

            existing_findings.append(row)
            seen_case_ids.add(case_id)
            seen_prompt_keys.add(prompt_key)
            arena_added += 1
            if item_org and item_org != "default":
                affected_orgs.add(item_org)

    # 4. Update summary metadata & persist JSON
    exported_at = datetime.now(timezone.utc).isoformat()
    total_attacks = sum(1 for f in existing_findings if int(f.get("label", 0)) == 1)
    total_caught = sum(
        1 for f in existing_findings if int(f.get("label", 0)) == 1 and bool(f.get("blocked", False))
    )
    total_benign = sum(1 for f in existing_findings if int(f.get("label", 0)) == 0)
    false_positives = sum(
        1 for f in existing_findings if int(f.get("label", 0)) == 0 and bool(f.get("blocked", False))
    )

    summary = dict(report.get("summary") or {})
    summary.update(
        {
            "total_attacks": total_attacks,
            "total_caught": total_caught,
            "overall_catch_rate": round(total_caught / total_attacks, 4) if total_attacks else 0.0,
            "total_benign": total_benign,
            "false_positives": false_positives,
            "overall_false_positive_rate": (
                round(false_positives / total_benign, 4) if total_benign else 0.0
            ),
        }
    )

    report["generated_at"] = exported_at
    report["summary"] = summary
    report.setdefault("per_target_metrics", {})
    report["findings"] = existing_findings
    report["last_feedback_export"] = {
        "exported_at": exported_at,
        "moderation_added": moderation_added,
        "arena_findings_added": arena_added,
        "duplicates_skipped": duplicates_skipped,
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(serialized)
        os.replace(tmp_name, target_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    # 5. Optional closed-loop per-client model retraining
    retrained: List[str] = []
    if retrain:
        from ares.training.client_model_trainer import (
            export_consolidated_ml_stats,
            train_client_models,
        )

        registry = ModelRegistry(registry_path=registry_path)
        known_clients = set(registry.list_clients().keys())

        if retrain_orgs:
            target_orgs = list(dict.fromkeys(retrain_orgs))
        elif org_id:
            target_orgs = [org_id]
        else:
            matched_known = sorted(oid for oid in affected_orgs if oid in known_clients)
            target_orgs = matched_known if matched_known else sorted(DEFAULT_CLIENT_PROFILES.keys())

        train_results = []
        for oid in target_orgs:
            train_results.append(
                train_client_models(
                    oid,
                    findings_path=target_path,
                    models_root=models_root,
                    registry_path=registry_path,
                )
            )
            retrained.append(oid)
        if train_results and models_root is None and registry_path is None:
            export_consolidated_ml_stats(train_results)

    return {
        "findings_path": str(target_path),
        "existing_count": existing_count,
        "moderation_added": moderation_added,
        "arena_findings_added": arena_added,
        "duplicates_skipped": duplicates_skipped,
        "total_findings": len(existing_findings),
        "retrained_orgs": retrained,
        "exported_at": exported_at,
    }
