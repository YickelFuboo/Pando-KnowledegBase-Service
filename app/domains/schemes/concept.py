from typing import List, Optional
from pydantic import BaseModel, Field


class ConceptItem(BaseModel):
    id: str
    concept_name: str
    description: Optional[str] = None
    doc_ids: List[str] = Field(default_factory=list, description="来源文档ID列表")
    kb_id: str
    document_names: Optional[List[str]] = Field(default=None, description="来源文档名称列表，与 doc_ids 顺序对应")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ListConceptsResponse(BaseModel):
    items: List[ConceptItem]
    total: int
    page: int
    page_size: int

