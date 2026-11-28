"""Add target provenance separation, baseline measurement tracking, and moderation counts to redteam campaigns.

Revision ID: 20261124_0007
Revises: 20261122_0006
Create Date: 2026-11-24
"""

from alembic import op
import sqlalchemy as sa

revision = "20261124_0007"
down_revision = "20261122_0006"
branch_labels = None
depends_on = None


def _column_names(inspector: sa.Inspector, table_name: str) -> set[str]:
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    camp_cols = _column_names(inspector, "redteam_campaigns")
    if camp_cols:
        with op.batch_alter_table("redteam_campaigns") as batch_op:
            if "target_mode" not in camp_cols:
                batch_op.add_column(
                    sa.Column("target_mode", sa.String(48), nullable=False, server_default="reference-simulator")
                )
            if "baseline_measured_count" not in camp_cols:
                batch_op.add_column(
                    sa.Column("baseline_measured_count", sa.Integer(), nullable=False, server_default="0")
                )
            if "baseline_unmeasured_count" not in camp_cols:
                batch_op.add_column(
                    sa.Column("baseline_unmeasured_count", sa.Integer(), nullable=False, server_default="0")
                )
            if "moderation_candidates_submitted" not in camp_cols:
                batch_op.add_column(
                    sa.Column("moderation_candidates_submitted", sa.Integer(), nullable=False, server_default="0")
                )
            if "metrics_by_target_mode" not in camp_cols:
                batch_op.add_column(
                    sa.Column("metrics_by_target_mode", sa.JSON(), nullable=False, server_default="{}")
                )

    case_cols = _column_names(inspector, "redteam_campaign_cases")
    if case_cols:
        with op.batch_alter_table("redteam_campaign_cases") as batch_op:
            if "target_mode" not in case_cols:
                batch_op.add_column(
                    sa.Column("target_mode", sa.String(48), nullable=False, server_default="reference-simulator")
                )
            if "prompt_sha256" not in case_cols:
                batch_op.add_column(
                    sa.Column("prompt_sha256", sa.String(64), nullable=False, server_default="")
                )
            if "baseline_measured" not in case_cols:
                batch_op.add_column(
                    sa.Column("baseline_measured", sa.Boolean(), nullable=False, server_default=sa.true())
                )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    case_cols = _column_names(inspector, "redteam_campaign_cases")
    if case_cols:
        with op.batch_alter_table("redteam_campaign_cases") as batch_op:
            for col in ("baseline_measured", "prompt_sha256", "target_mode"):
                if col in case_cols:
                    batch_op.drop_column(col)

    camp_cols = _column_names(inspector, "redteam_campaigns")
    if camp_cols:
        with op.batch_alter_table("redteam_campaigns") as batch_op:
            for col in (
                "metrics_by_target_mode",
                "moderation_candidates_submitted",
                "baseline_unmeasured_count",
                "baseline_measured_count",
                "target_mode",
            ):
                if col in camp_cols:
                    batch_op.drop_column(col)
