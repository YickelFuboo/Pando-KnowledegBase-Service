"""documents add web_url

Revision ID: 007
Revises: 006
Create Date: 2026-03-11 00:00:03.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "007"
down_revision: Union[str, Sequence[str], None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("documents")}
    if "web_url" not in columns:
        op.add_column(
            "documents",
            sa.Column("web_url", sa.String(length=1024), nullable=True, comment="网页来源URL"),
        )
        op.create_index(op.f("ix_documents_web_url"), "documents", ["web_url"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("documents")}
    if "web_url" in columns:
        op.drop_index(op.f("ix_documents_web_url"), table_name="documents")
        op.drop_column("documents", "web_url")
