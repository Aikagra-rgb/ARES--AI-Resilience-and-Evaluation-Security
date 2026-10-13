"""
Tests for ARES Alembic Migration Pipeline and Schema Integrity.

Verifies:
1. Alembic configuration resolves script locations reliably.
2. Migration chain is unbroken (single linear history from base to head).
3. Exactly one head exists without diverging branches.
4. All declared SQLAlchemy models are mapped to Base.metadata.
"""

from pathlib import Path
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.database import Base
from app.models import (
    Organization,
    User,
    Project,
    Policy,
    TestRun,
    Finding,
    Evidence,
    ArenaSubmission,
    AuditLog,
)


def test_alembic_single_head_and_linear_history():
    ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    assert ini_path.exists(), f"alembic.ini not found at {ini_path}"

    config = Config(str(ini_path))
    script = ScriptDirectory.from_config(config)

    # 1. Exactly one head
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected single migration head, found multiple: {heads}"
    assert heads[0] == "20261004_0004"

    # 2. Check linear chain from base to head
    revisions = list(script.walk_revisions(base="base", head=heads[0]))
    rev_ids = [r.revision for r in revisions]

    expected_chain = [
        "20261004_0004",
        "20260904_0003",
        "20260904_0002",
        "20260904_0001",
    ]
    assert rev_ids == expected_chain, f"Migration chain mismatch: {rev_ids}"


def test_sqlalchemy_models_registered_in_metadata():
    tables = Base.metadata.tables.keys()

    expected_tables = {
        "organizations",
        "users",
        "projects",
        "policies",
        "test_runs",
        "findings",
        "evidence",
        "arena_submissions",
        "audit_logs",
    }

    assert expected_tables.issubset(tables), f"Missing tables in metadata: {expected_tables - set(tables)}"
