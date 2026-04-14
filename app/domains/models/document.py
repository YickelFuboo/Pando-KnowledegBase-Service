from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.infrastructure.database.models_base import Base
from app.rag_core.utils import ParserType
from app.constants.common import FileType, FileSource


class ProcessStatus(str, Enum):
    """文档处理状态枚举"""
    INIT = "init"
    CHUNKING = "chunking"
    RAPTORING = "raptoring"
    GRAPHING = "graphing"
    PARSED = "parsed"
    FAILED = "failed"

class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, comment="文档ID")
    kb_id = Column(String(36), ForeignKey("knowledgebase.id"), nullable=False, comment="知识库ID")
    name = Column(String(255), nullable=False, comment="文档名称")
    description = Column(Text, comment="文档描述")
    type = Column(String(20), nullable=False, default=FileType.PDF, comment="文档类型")
    suffix = Column(String(10), comment="文件扩展名")
    file_id = Column(String(500), comment="文件唯一标识符")
    size = Column(Integer, default=0, comment="文件大小(字节)")
    parser_id = Column(String(50), nullable=False, default=ParserType.GENERAL, comment="解析器类型")
    parser_config = Column(JSON, default={}, comment="解析器配置(JSON)")
    meta_fields = Column(JSON, default={}, comment="文档元数据字段(JSON)")
    thumbnail_id = Column(String(500), comment="缩略图ID")
    source_type = Column(String(20), default=FileSource.UPLOAD, comment="文件来源")
    web_url = Column(String(1024), nullable=True, index=True, comment="网页来源URL")
    process_status = Column(
        SQLEnum(ProcessStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=ProcessStatus.INIT,
        comment="文档处理状态"
    )
    created_by = Column(String(36), nullable=False, comment="创建者ID")

    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    # 关联关系
    knowledgebase = relationship("KB", back_populates="documents")
