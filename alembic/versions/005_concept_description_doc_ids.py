"""add concept description and doc_ids, unique (kb_id, concept_name)

Revision ID: 005
Revises: 004
Create Date: 2026-03-11 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text, inspect

revision: str = "005"
down_revision: Union[str, Sequence[str], None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    dialect_name = conn.dialect.name
    inspector = inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("kb_concepts")}

    if "description" not in columns:
        op.add_column("kb_concepts", sa.Column("description", sa.Text(), nullable=True, comment="概念描述"))
        columns.add("description")
    if "doc_ids" not in columns:
        op.add_column("kb_concepts", sa.Column("doc_ids", sa.JSON(), nullable=True, comment="来源文档ID列表"))
        columns.add("doc_ids")
    if "updated_at" not in columns:
        op.add_column("kb_concepts", sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=True, comment="更新时间"))
        columns.add("updated_at")

    if "doc_ids" in columns:
        if dialect_name == "sqlite":
            conn.execute(text("UPDATE kb_concepts SET doc_ids = json_array(doc_id), updated_at = created_at"))
        elif dialect_name == "postgresql":
            conn.execute(text("UPDATE kb_concepts SET doc_ids = jsonb_build_array(doc_id), updated_at = COALESCE(created_at, now())"))
        else:
            conn.execute(text("UPDATE kb_concepts SET doc_ids = JSON_ARRAY(doc_id), updated_at = COALESCE(created_at, now())"))

    if dialect_name == "sqlite":
        # SQLite 不支持直接使用 ALTER COLUMN / DROP COLUMN 语法，使用 batch_alter_table 让 Alembic 重建表
        with op.batch_alter_table("kb_concepts") as batch_op:
            if "doc_ids" in columns:
                batch_op.alter_column("doc_ids", existing_type=sa.JSON(), nullable=False)
            if "updated_at" in columns:
                batch_op.alter_column("updated_at", existing_type=sa.DateTime(), nullable=False, existing_nullable=True)
            if "doc_id" in columns:
                # 先删索引再删列，避免 SQLite 外键定义引用已删除列
                batch_op.drop_index("ix_kb_concepts_doc_id")
                batch_op.drop_column("doc_id")
    else:
        if "doc_ids" in columns:
            op.alter_column("kb_concepts", "doc_ids", nullable=False)
        if "updated_at" in columns:
            op.alter_column("kb_concepts", "updated_at", nullable=False, existing_nullable=True)
        if "doc_id" in columns:
            op.drop_index("ix_kb_concepts_doc_id", table_name="kb_concepts")
            op.drop_column("kb_concepts", "doc_id")

    # 在 SQLite 上不能直接 ALTER TABLE ADD CONSTRAINT，这里用唯一索引代替；
    # 其他数据库继续使用真正的 UNIQUE CONSTRAINT。
    if dialect_name == "sqlite":
        op.create_index(
            "uq_kb_concepts_kb_id_concept_name",
            "kb_concepts",
            ["kb_id", "concept_name"],
            unique=True,
        )
    else:
        op.create_unique_constraint(
            "uq_kb_concepts_kb_id_concept_name",
            "kb_concepts",
            ["kb_id", "concept_name"],
        )


def downgrade() -> None:
    op.drop_constraint("uq_kb_concepts_kb_id_concept_name", "kb_concepts", type_="unique")
    op.add_column("kb_concepts", sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id"), nullable=True))
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        conn.execute(text("UPDATE kb_concepts SET doc_id = json_extract(doc_ids, '$[0]')"))
    elif conn.dialect.name == "postgresql":
        conn.execute(text("UPDATE kb_concepts SET doc_id = doc_ids->>0"))
    else:
        conn.execute(text("UPDATE kb_concepts SET doc_id = JSON_UNQUOTE(JSON_EXTRACT(doc_ids, '$[0]'))"))
    op.alter_column("kb_concepts", "doc_id", nullable=False)
    op.create_index("ix_kb_concepts_doc_id", "kb_concepts", ["doc_id"])
    op.drop_column("kb_concepts", "updated_at")
    op.drop_column("kb_concepts", "doc_ids")
    op.drop_column("kb_concepts", "description")
