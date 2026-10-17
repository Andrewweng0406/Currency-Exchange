"""Add immutable forward-validation snapshots and outcomes.

Revision ID: 20261016_0002
Revises: 20260826_0001
"""
from alembic import op
import sqlalchemy as sa

revision = "20261016_0002"
down_revision = "20260826_0001"
branch_labels = None
depends_on = None


def upgrade():
    # Freeze this revision's DDL rather than importing future application models.
    # Development startup may already have created the same additive tables.
    offline = op.get_context().as_sql
    inspector = None if offline else sa.inspect(op.get_bind())
    if offline or not inspector.has_table("forward_snapshots"):
        op.create_table(
            "forward_snapshots",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("protocol_hash", sa.String(64), nullable=False),
            sa.Column("protocol_json", sa.Text(), nullable=False),
            sa.Column("origin_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("horizon", sa.String(10), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("issued_at_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("recorded_at_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column("payload_hash", sa.String(64), nullable=False),
            sa.UniqueConstraint("protocol_hash", "origin_utc", "horizon", name="uq_forward_origin"),
        )
        op.create_index("ix_forward_snapshots_protocol_hash", "forward_snapshots", ["protocol_hash"])
        op.create_index("ix_forward_snapshots_origin_utc", "forward_snapshots", ["origin_utc"])
    if offline or not inspector.has_table("forward_outcomes"):
        op.create_table(
            "forward_outcomes",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("snapshot_id", sa.Integer(), sa.ForeignKey("forward_snapshots.id"), nullable=False),
            sa.Column("target_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("evaluated_at_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("close", sa.Float(), nullable=False),
            sa.Column("realized_return", sa.Float(), nullable=False),
            sa.Column("target_up", sa.Integer(), nullable=False),
            sa.Column("evidence_json", sa.Text(), nullable=False),
        )
        op.create_index("ix_forward_outcomes_snapshot_id", "forward_outcomes", ["snapshot_id"], unique=True)


def downgrade():
    op.drop_table("forward_outcomes")
    op.drop_table("forward_snapshots")
