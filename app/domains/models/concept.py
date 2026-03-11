from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.sql import func
from app.infrastructure.database.models_base import Base


class KBConcept(Base):
    """知识库概念表：按知识库维度存储从文档切片中提取的名词概念，含描述，支持多文档关联"""
    __tablename__ = "kb_concepts"
    __table_args__ = (UniqueConstraint("kb_id", "concept_name", name="uq_kb_concepts_kb_id_concept_name"),)

    id = Column(String(36), primary_key=True, comment="概念记录ID")
    kb_id = Column(String(32), ForeignKey("knowledgebase.id", ondelete="CASCADE"), nullable=False, index=True, comment="知识库ID")
    concept_name = Column(String(256), nullable=False, index=True, comment="概念名称")
    description = Column(Text, nullable=True, comment="概念描述")
    doc_ids = Column(JSON, nullable=False, default=lambda: [], comment="来源文档ID列表")
    created_at = Column(DateTime, nullable=False, server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now(), comment="更新时间")
