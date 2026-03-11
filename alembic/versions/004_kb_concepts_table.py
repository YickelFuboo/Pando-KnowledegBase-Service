"""add kb_concepts table for concept extraction

Revision ID: 004
Revises: 003
Create Date: 2026-03-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, Sequence[str], None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kb_concepts",
        sa.Column("id", sa.String(36), primary_key=True, comment="概念记录ID"),
        sa.Column("kb_id", sa.String(32), sa.ForeignKey("knowledgebase.id"), nullable=False, comment="知识库ID"),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id"), nullable=False, comment="来源文档ID"),
        sa.Column("concept_name", sa.String(256), nullable=False, comment="概念名称"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, comment="创建时间"),
    )
    op.create_index("ix_kb_concepts_kb_id", "kb_concepts", ["kb_id"])
    op.create_index("ix_kb_concepts_doc_id", "kb_concepts", ["doc_id"])
    op.create_index("ix_kb_concepts_concept_name", "kb_concepts", ["concept_name"])


def downgrade() -> None:
    op.drop_index("ix_kb_concepts_concept_name", table_name="kb_concepts")
    op.drop_index("ix_kb_concepts_doc_id", table_name="kb_concepts")
    op.drop_index("ix_kb_concepts_kb_id", table_name="kb_concepts")
    op.drop_table("kb_concepts")
