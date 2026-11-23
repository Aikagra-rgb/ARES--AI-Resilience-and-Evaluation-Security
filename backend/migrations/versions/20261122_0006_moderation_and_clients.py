"""Persistent moderation queue items and durable client tenant registry metadata.

Revision ID: 20261122_0006
Revises: 20261120_0005
Create Date: 2026-11-22
"""

from alembic import op
import sqlalchemy as sa

revision = "20261122_0006"
down_revision = "20261120_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "moderation_queue_items" not in existing_tables:
        op.create_table(
            "moderation_queue_items",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("attack_text", sa.Text(), nullable=False),
            sa.Column("category", sa.String(80), nullable=False, server_default="jailbreak"),
            sa.Column("severity", sa.String(32), nullable=False, server_default="high"),
            sa.Column("source", sa.String(120), nullable=False, server_default="redteam_probe"),
            sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
            sa.Column("domain", sa.String(64), nullable=False, server_default="general"),
            sa.Column("org_id", sa.String(128), nullable=True),
            sa.Column("target_prompt_hash", sa.String(128), nullable=False, server_default="moderated_sample"),
            sa.Column("item_metadata", sa.JSON(), nullable=False),
            sa.Column("submitted_by", sa.String(160), nullable=True),
            sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("reviewed_by", sa.String(160), nullable=True),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("reviewer_notes", sa.Text(), nullable=True),
            sa.Column("deduplication_score", sa.Float(), nullable=True),
            sa.Column("duplicate_of_id", sa.String(128), nullable=True),
            sa.Column("poisoning_flags", sa.JSON(), nullable=False),
            sa.Column("indexed_point_id", sa.String(128), nullable=True),
        )
        op.create_index("ix_moderation_queue_items_status", "moderation_queue_items", ["status"])
        op.create_index("ix_moderation_queue_items_category", "moderation_queue_items", ["category"])
        op.create_index("ix_moderation_queue_items_domain", "moderation_queue_items", ["domain"])
        op.create_index("ix_moderation_queue_items_org_id", "moderation_queue_items", ["org_id"])
        op.create_index("ix_moderation_queue_items_submitted_at", "moderation_queue_items", ["submitted_at"])

    if "client_tenants" not in existing_tables:
        op.create_table(
            "client_tenants",
            sa.Column("org_id", sa.String(128), primary_key=True),
            sa.Column("display_name", sa.String(160), nullable=False),
            sa.Column("domain", sa.String(64), nullable=False, server_default="general"),
            sa.Column("application_name", sa.String(160), nullable=False),
            sa.Column("canary_token", sa.String(256), nullable=True),
            sa.Column("intent_model", sa.String(512), nullable=True),
            sa.Column("compression_adapter", sa.String(512), nullable=True),
            sa.Column("version", sa.String(64), nullable=False, server_default="untrained"),
            sa.Column("fidelity_thresholds", sa.JSON(), nullable=False),
            sa.Column("metrics", sa.JSON(), nullable=False),
            sa.Column("target_endpoint_url", sa.String(512), nullable=True),
            sa.Column("target_auth_header", sa.String(512), nullable=True),
            sa.Column("target_request_format", sa.String(64), nullable=False, server_default="openai_chat"),
            sa.Column("target_model", sa.String(200), nullable=True),
            sa.Column("api_key_hash", sa.String(128), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_client_tenants_domain", "client_tenants", ["domain"])
        op.create_index("ix_client_tenants_application_name", "client_tenants", ["application_name"])
        op.create_index("ix_client_tenants_api_key_hash", "client_tenants", ["api_key_hash"])
        op.create_index("ix_client_tenants_updated_at", "client_tenants", ["updated_at"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    if "client_tenants" in existing_tables:
        op.drop_table("client_tenants")
    if "moderation_queue_items" in existing_tables:
        op.drop_table("moderation_queue_items")
