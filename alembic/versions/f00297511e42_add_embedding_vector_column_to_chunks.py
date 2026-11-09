"""add embedding vector column to chunks

Revision ID: f00297511e42
Revises: abdea56e718a
Create Date: 2026-11-08 15:58:05.742154

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'f00297511e42'
down_revision: Union[str, Sequence[str], None] = 'abdea56e718a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("chunks", sa.Column("embedding", Vector(1536), nullable=True))
    op.add_column("chunks", sa.Column("embedding_model", sa.String(length=100), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
