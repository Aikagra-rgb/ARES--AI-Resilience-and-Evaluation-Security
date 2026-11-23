"""
CLI Bridge: Export Approved Moderation Queue Items & Completed Arena Findings to Training Data.

Merges human-approved moderation items and completed Arena `TestRun.findings` into
`backend/data/redteam_findings.json` with deduplication, and optionally retrains
per-client models via `train_client_models`.

Usage:
    python backend/scripts/export_feedback_to_training.py
    python backend/scripts/export_feedback_to_training.py --org org_acme_fintech --retrain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ares.moderation.feedback_export import (  # noqa: E402
    DEFAULT_FINDINGS_PATH,
    export_feedback_to_training,
)

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge approved moderation items and Arena findings into redteam_findings.json."
    )
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=DEFAULT_FINDINGS_PATH,
        help="Target redteam_findings.json file path.",
    )
    parser.add_argument(
        "--org",
        dest="org_id",
        default=None,
        help="Optional organization ID filter.",
    )
    parser.add_argument(
        "--skip-moderation",
        action="store_true",
        help="Skip exporting approved ModerationQueueItem records.",
    )
    parser.add_argument(
        "--skip-arena",
        action="store_true",
        help="Skip exporting completed Arena TestRun.findings records.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Trigger per-client train_client_models after merging feedback.",
    )
    parser.add_argument(
        "--retrain-orgs",
        nargs="*",
        default=None,
        help="Optional explicit list of org IDs to retrain when --retrain is set.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)
    result = export_feedback_to_training(
        findings_path=args.findings_path,
        org_id=args.org_id,
        include_moderation=not args.skip_moderation,
        include_arena_findings=not args.skip_arena,
        retrain=args.retrain,
        retrain_orgs=args.retrain_orgs,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"Exported feedback to {result['findings_path']}: "
            f"+{result['moderation_added']} moderation, "
            f"+{result['arena_findings_added']} arena, "
            f"{result['duplicates_skipped']} skipped, "
            f"total={result['total_findings']}, "
            f"retrained={result['retrained_orgs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
