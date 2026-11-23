"""
Auditable Per-Organization Domain Dataset Splitter.

Materializes stratified, leak-free `train.jsonl`, `val.jsonl`, `test.jsonl` splits
plus a SHA-256 `manifest.json` under `backend/data/splits/{org_id}/`.

Usage:
    python backend/scripts/split_dataset_by_domain.py --org all
    python backend/scripts/split_dataset_by_domain.py --org org_acme_fintech --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ares.intent.taxonomy import PromptIntent  # noqa: E402
from ares.training.distilbert_intent_trainer import (  # noqa: E402
    DEFAULT_FINDINGS_PATH,
    DEFAULT_INTENT_DATASET_PATH,
    _DOMAIN_DATASET_KEYWORDS,
    _infer_domain,
    _normalize_threat_label,
)
from ares.training.model_registry import (  # noqa: E402
    DEFAULT_CLIENT_PROFILES,
    ModelRegistry,
    _SAFE_ORG_ID_RE,  # noqa: F401
    validate_org_id,
)

logger = logging.getLogger(__name__)

DEFAULT_SPLITS_ROOT = _BACKEND_DIR / "data" / "splits"


def _validate_org_id(org_id: str) -> str:
    return validate_org_id(org_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_org_records(
    org_id: str,
    domain: str,
    intent_dataset_path: Path,
    findings_path: Path,
) -> List[Dict[str, Any]]:
    """Collect deduplicated domain-matched benign intent records and org/global threat records."""
    keywords = _DOMAIN_DATASET_KEYWORDS.get(domain, _DOMAIN_DATASET_KEYWORDS["general"])
    valid_intents = {i.value for i in PromptIntent}
    seen_texts: set[str] = set()
    records: List[Dict[str, Any]] = []

    # 1. Domain-filtered intent records
    if intent_dataset_path.exists():
        with intent_dataset_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(rec.get("prompt") or "").strip()
                intent = str(rec.get("intent") or "").strip()
                rec_domain = str(rec.get("domain") or "general").strip().lower()
                if not text or intent not in valid_intents:
                    continue
                if not any(kw in rec_domain for kw in keywords):
                    continue
                norm_key = text.lower()
                if norm_key in seen_texts:
                    continue
                seen_texts.add(norm_key)
                records.append(
                    {
                        "id": f"intent_{idx}",
                        "org_id": org_id,
                        "domain": domain,
                        "source_domain": rec_domain,
                        "text": text,
                        "label": intent,
                        "threat_label": "benign",
                        "label_type": "intent",
                        "source": intent_dataset_path.name,
                    }
                )

    # 2. Red-team / moderation / Arena findings (prioritizing org-matching entries)
    if findings_path.exists():
        try:
            raw = json.loads(findings_path.read_text(encoding="utf-8"))
            findings = raw.get("findings", []) if isinstance(raw, dict) else []
        except Exception:
            findings = []

        for idx, item in enumerate(findings):
            text = str(item.get("prompt") or "").strip()
            if not text:
                continue
            item_org = str(item.get("organization_id") or "default").strip()
            # Include org-specific findings and general/default findings; exclude other tenants' private findings
            if item_org not in (org_id, "default", ""):
                continue
            norm_key = text.lower()
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)
            is_attack = int(item.get("label", 0)) == 1
            threat_lbl = (
                _normalize_threat_label(item.get("attack_category") or item.get("matched_technique"))
                if is_attack
                else "benign"
            )
            records.append(
                {
                    "id": str(item.get("case_id") or f"finding_{idx}"),
                    "org_id": org_id,
                    "domain": domain,
                    "source_domain": domain,
                    "text": text,
                    "label": threat_lbl,
                    "threat_label": threat_lbl,
                    "label_type": "threat" if is_attack else "benign_control",
                    "source": str(item.get("source") or findings_path.name),
                }
            )

    return records


def _stratified_split(
    records: List[Dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically partition records into stratified train/val/test lists."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_label[str(rec["label"])].append(rec)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for label in sorted(by_label.keys()):
        group = list(by_label[label])
        rng.shuffle(group)
        n = len(group)
        if n == 1:
            train_rows.extend(group)
        elif n == 2:
            train_rows.append(group[0])
            test_rows.append(group[1])
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            n_train = n - n_val - n_test
            train_rows.extend(group[:n_train])
            val_rows.extend(group[n_train : n_train + n_val])
            test_rows.extend(group[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def split_dataset_for_org(
    org_id: str,
    *,
    domain: Optional[str] = None,
    output_root: Path = DEFAULT_SPLITS_ROOT,
    intent_dataset_path: Path = DEFAULT_INTENT_DATASET_PATH,
    findings_path: Path = DEFAULT_FINDINGS_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Materialize auditable train/val/test JSONL files and SHA-256 manifest for `org_id`."""
    org_id = _validate_org_id(org_id)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6 or min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError(
            f"Split ratios must be positive and sum to 1.0 (got {train_ratio}, {val_ratio}, {test_ratio})"
        )

    resolved_domain = _infer_domain(org_id, domain)
    records = _collect_org_records(
        org_id=org_id,
        domain=resolved_domain,
        intent_dataset_path=Path(intent_dataset_path),
        findings_path=Path(findings_path),
    )
    train_rows, val_rows, test_rows = _stratified_split(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    root_resolved = Path(output_root).resolve()
    org_dir = (root_resolved / org_id).resolve()
    if root_resolved not in org_dir.parents:
        raise ValueError(f"Organization split directory escaped output root: {org_id}")
    org_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }
    split_manifests: Dict[str, Dict[str, Any]] = {}
    text_sets: Dict[str, set[str]] = {}

    for split_name, rows in split_map.items():
        filename = f"{split_name}.jsonl"
        file_path = org_dir / filename
        with file_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        label_dist = dict(sorted(Counter(str(r["label"]) for r in rows).items()))
        text_sets[split_name] = {str(r["text"]).strip().lower() for r in rows}
        split_manifests[split_name] = {
            "filename": filename,
            "path": str(file_path),
            "record_count": len(rows),
            "sha256": _sha256_file(file_path),
            "label_distribution": label_dist,
        }

    # Verify zero text overlap across train, val, and test splits
    overlap_tv = text_sets["train"] & text_sets["val"]
    overlap_tt = text_sets["train"] & text_sets["test"]
    overlap_vt = text_sets["val"] & text_sets["test"]
    leakage_free = len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0

    sources_meta: Dict[str, Any] = {}
    for key, src_path in (
        ("intent_dataset", Path(intent_dataset_path)),
        ("redteam_findings", Path(findings_path)),
    ):
        sources_meta[key] = {
            "path": str(src_path),
            "exists": src_path.exists(),
            "sha256": _sha256_file(src_path) if src_path.exists() else None,
        }

    manifest = {
        "org_id": org_id,
        "domain": resolved_domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "total_records": len(records),
        "leakage_check_passed": leakage_free,
        "sources": sources_meta,
        "splits": split_manifests,
    }
    manifest_path = org_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize auditable per-org train/val/test JSONL splits with SHA-256 manifest."
    )
    parser.add_argument(
        "--org",
        default="all",
        help="Target organization ID (e.g., org_acme_fintech) or 'all' for all registered clients.",
    )
    parser.add_argument("--domain", default=None, help="Optional explicit domain override.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPLITS_ROOT,
        help="Root output directory for per-org splits.",
    )
    parser.add_argument(
        "--intent-dataset",
        type=Path,
        default=DEFAULT_INTENT_DATASET_PATH,
        help="Path to intent dataset JSONL.",
    )
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=DEFAULT_FINDINGS_PATH,
        help="Path to redteam_findings.json.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print full JSON manifest summary.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.org.lower() == "all":
        clients = ModelRegistry().list_clients()
        org_ids = sorted(clients.keys()) if clients else sorted(DEFAULT_CLIENT_PROFILES.keys())
    else:
        org_ids = [args.org]

    manifests = []
    for oid in org_ids:
        manifest = split_dataset_for_org(
            oid,
            domain=args.domain,
            output_root=args.output_root,
            intent_dataset_path=args.intent_dataset,
            findings_path=args.findings_path,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        manifests.append(manifest)

    if args.json:
        print(json.dumps({"orgs": [m["org_id"] for m in manifests], "manifests": manifests}, indent=2))
    else:
        for m in manifests:
            s = m["splits"]
            print(
                f"[{m['org_id']}] ({m['domain']}) total={m['total_records']} "
                f"train={s['train']['record_count']} val={s['val']['record_count']} test={s['test']['record_count']} "
                f"leakage_free={m['leakage_check_passed']} -> {m['manifest_path']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
