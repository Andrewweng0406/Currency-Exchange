"""initial schema

Revision ID: 20260826_0001
Revises:
Create Date: 2026-08-26 00:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260826_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The application models are the source of truth. For existing SQLite local
    # databases this migration can be stamped; for new production databases,
    # SQLAlchemy creates tables during app startup until stricter migrations are
    # introduced. This baseline records the schema version without destructive DDL.
    pass


def downgrade() -> None:
    pass
