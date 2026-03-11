from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.infrastructure.database import get_db
from app.domains.services.kb_service import KBService
from app.domains.services.concept_service import ConceptService
from app.domains.services.doc_service import DocumentService
from app.domains.schemes.concept import ConceptItem, ListConceptsResponse
from app.domains.models import KB, KBConcept
from sqlalchemy import select


router = APIRouter()


@router.get("", response_model=ListConceptsResponse)
async def list_concepts(
    kb_ids: List[str] = Query(..., description="知识库ID列表，可重复传参"),
    keyword: Optional[str] = Query(None, description="概念名称关键词，模糊查询"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    user_id: str = Query(..., description="用户ID"),
    session: AsyncSession = Depends(get_db),
):
    """按多个知识库分页查询概念列表；支持 keyword 对概念名称模糊查询。"""
    items, total = await ConceptService.get_by_kb_ids(
        db_session=session, kb_ids=kb_ids, keyword=keyword, limit=page_size, page=page
    )
    total = total if total is not None else len(items)
    all_doc_ids = []
    for x in items:
        all_doc_ids.extend(list(x.doc_ids) if x.doc_ids else [])
    all_doc_ids = list(dict.fromkeys(all_doc_ids))

    docs = await DocumentService.get_documents_by_ids(session, all_doc_ids) if all_doc_ids else []
    doc_id_to_name = {d.id: d.name for d in docs}
    out = []
    for x in items:
        doc_ids = list(x.doc_ids) if x.doc_ids else []
        document_names = [doc_id_to_name.get(did, "") for did in doc_ids]
        out.append(
            ConceptItem(
                id=x.id,
                concept_name=x.concept_name,
                description=x.description,
                doc_ids=doc_ids,
                kb_id=x.kb_id,
                document_names=document_names,
                created_at=x.created_at.isoformat() if x.created_at else None,
                updated_at=x.updated_at.isoformat() if x.updated_at else None,
            )
        )
    return ListConceptsResponse(items=out, total=total, page=page, page_size=page_size)
