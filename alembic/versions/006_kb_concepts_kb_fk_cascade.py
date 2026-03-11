"""kb_concepts: kb_id fk add ondelete cascade

Revision ID: 006
Revises: 005
Create Date: 2026-03-11 00:00:02.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "006"
down_revision: Union[str, Sequence[str], None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = inspect(conn)
    fks = insp.get_foreign_keys("kb_concepts")
    kb_fk_names = []
    for fk in fks:
        if fk.get("referred_table") == "knowledgebase" and fk.get("constrained_columns") == ["kb_id"]:
            if fk.get("name"):
                kb_fk_names.append(fk["name"])

    with op.batch_alter_table("kb_concepts") as batch:
        for name in kb_fk_names:
            batch.drop_constraint(name, type_="foreignkey")
        batch.create_foreign_key(
            "fk_kb_concepts_kb_id_knowledgebase",
            "knowledgebase",
            ["kb_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = inspect(conn)
    fks = insp.get_foreign_keys("kb_concepts")
    names = []
    for fk in fks:
        if fk.get("referred_table") == "knowledgebase" and fk.get("constrained_columns") == ["kb_id"]:
            if fk.get("name"):
                names.append(fk["name"])

    with op.batch_alter_table("kb_concepts") as batch:
        for name in names:
            batch.drop_constraint(name, type_="foreignkey")
        batch.create_foreign_key(
            "fk_kb_concepts_kb_id_knowledgebase",
            "knowledgebase",
            ["kb_id"],
            ["id"],
        )

