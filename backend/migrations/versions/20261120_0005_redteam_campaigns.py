"""Persistent multi-tenant Red-Team campaigns and case verdicts.

Revision ID: 20261120_0005
Revises: 20261004_0004
Create Date: 2026-11-20
"""
from alembic import op
import sqlalchemy as sa

revision = "20261120_0005"
down_revision = "20261004_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "redteam_campaigns" not in existing_tables:
        op.create_table(
            "redteam_campaigns",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("run_id", sa.String(64), nullable=False, unique=True),
            sa.Column("org_id", sa.String(128), nullable=False),
            sa.Column("use_mock_llm", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("total_attacks", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("attacks_blocked", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("catch_rate", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("total_benign", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("false_positives", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("false_positive_rate", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("baseline_breaches", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("latency_p50_ms", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("latency_p99_ms", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("findings_exported", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("qdrant_vectors_upserted", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("retrained_orgs", sa.JSON(), nullable=False),
            sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_redteam_campaigns_run_id", "redteam_campaigns", ["run_id"], unique=True)
        op.create_index("ix_redteam_campaigns_org_executed", "redteam_campaigns", ["org_id", "executed_at"])

    if "redteam_campaign_cases" not in existing_tables:
        op.create_table(
            "redteam_campaign_cases",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "campaign_id",
                sa.String(36),
                sa.ForeignKey("redteam_campaigns.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("case_id", sa.String(128), nullable=False),
            sa.Column("org_id", sa.String(128), nullable=False),
            sa.Column("target_name", sa.String(128), nullable=False),
            sa.Column("attack_category", sa.String(80), nullable=False),
            sa.Column("query_preview", sa.String(256), nullable=False),
            sa.Column("expected_blocked", sa.Boolean(), nullable=False),
            sa.Column("baseline_compromised", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("protected_blocked", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("action", sa.String(32), nullable=False),
            sa.Column("matched_technique", sa.String(120), nullable=True),
            sa.Column("canary_leaked", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        )
        op.create_index("ix_redteam_cases_campaign_id", "redteam_campaign_cases", ["campaign_id"])
        op.create_index("ix_redteam_cases_org_id", "redteam_campaign_cases", ["org_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "redteam_campaign_cases" in existing_tables:
        op.drop_table("redteam_campaign_cases")
    if "redteam_campaigns" in existing_tables:
        op.drop_table("redteam_campaigns")
