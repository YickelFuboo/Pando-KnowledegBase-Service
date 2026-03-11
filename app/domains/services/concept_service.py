import json
import re
import uuid
from typing import Any,Dict,List,Optional,Tuple
from pydantic import BaseModel
from sqlalchemy import delete,func,select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domains.models import KB,KBConcept
from app.infrastructure.llms import llm_factory


MAX_CONCEPTS = 50

_CONCEPT_EXTRACT_SYS_PROMPT = """You are a text analyst. Extract ONLY document-specific concepts; do NOT extract common knowledge or generic terms.

Task:
From the given text, extract noun concepts that are specific to this document (e.g. defined here, given special meaning here, or uniquely referred to here), and write a short description for each. The goal is to capture document-specific knowledge for retrieval and QA, distinct from general knowledge.

Exclude (do NOT extract):
- Common knowledge: Any concept that the model already knows, is widely known, or can be found in dictionaries or encyclopedias. This applies to any domain (legal, contract, industry terms, etc.)—if it is common knowledge, do not extract it.
- Generic nouns that are only listed or mentioned without being defined or explained in this document.

Extract only (document-specific):
- Concepts that are explicitly defined, explained, or given a special meaning in this document (e.g. document-invented terms, agreement-specific definitions).
- Proper names that appear in this document: company names, product names, project names, person names, place names, file names, clause IDs, etc.
- Abbreviations, codes, or internal jargon that have a specific meaning in this document.
- If the text is entirely generic and has no document-specific concepts above, return an empty list.

Rules:
1) Extract noun concepts only; do not extract verbs or adjective phrases.
2) Concept name and description must be in the same language as the input text.
3) Prefer fewer, precise concepts. When unsure whether a concept is document-specific, do not extract it.
4) Output MUST be valid JSON only, with exactly this structure:
{
  "concepts": [
    {"name": "Concept name", "description": "Short description"}
  ]
}
5) At most 50 concepts. If there are no document-specific concepts, return {"concepts": []}.
"""

_CONCEPT_MULTI_MERGE_DECISION_SYS_PROMPT = """You are a concept deduplication and merge decision maker.

Task:
Given a list of potentially similar concepts (each has id/name/description), decide:
- Which concepts should be merged into the same concept (merge all / merge some / or merge none)
- Concepts that are NOT merged must be kept AS-IS (do NOT change their name/description)

Merge rules:
1) Merge ONLY when concepts represent the same concept / synonym / the same entity.
   Do NOT merge hypernym-hyponym relations, containment, or merely related-but-different concepts.
2) For each merged group, produce ONE merged concept with a canonicalized name (minor normalization is allowed),
   and a concise description summarizing the merged group.
3) Do NOT merge just to merge. If unsure, do NOT merge.

Output requirements:
Output MUST be JSON and JSON only. It MUST match this schema:
{
  "merge": [
    {
      "source_ids": ["id1","id2"],
      "name": "Merged concept name",
      "description": "Merged description"
    }
  ],
  "keep_ids": ["id3","id4"]
}

Notes:
- Each merge item merges the concepts in source_ids into ONE concept; source_ids must have at least 2 ids.
- keep_ids are concepts that remain unchanged.
- Every input concept id MUST appear exactly once either in some merge.source_ids or in keep_ids (no duplicates, no omissions).
"""


def _norm_doc_ids(doc_ids: Any) -> List[str]:
    if doc_ids is None:
        return []
    if isinstance(doc_ids, list):
        return [str(x) for x in doc_ids if x]
    return []

def _get_chat_content(chat_result) -> str:
    """从 chat 返回值 (ChatResponse, int) 中取出 content 并去除 markdown 代码块与 </think> 标签。"""
    if isinstance(chat_result, tuple):
        resp = chat_result[0]
    else:
        resp = chat_result
    if resp is None:
        return ""
    if isinstance(resp, str):
        raw = resp
    else:
        raw = getattr(resp, "content", None)
    if raw is None:
        return ""
    raw = str(raw).strip()
    raw = re.sub(r"^.*</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


def _safe_json_loads(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"(\{[\\s\\S]*\})", s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

class ConceptMerge(BaseModel):
    source_ids: List[str]   
    name: str
    description: str

class ConceptService:
    """知识库概念服务：概念提取结果的存储、合并与查询"""

    _NEW_CONCEPT_TEMP_ID = "__new__"

    @staticmethod
    async def _create_chat_model(db_session: AsyncSession, kb_id: str):
        return llm_factory.create_model()

    @staticmethod
    def parse_extract_json(raw: str) -> List[Tuple[str, str]]:
        obj = _safe_json_loads(raw)
        if not obj or not isinstance(obj, dict):
            return []

        concepts = obj.get("concepts", [])
        if not isinstance(concepts, list):
            return []
        
        result: List[Tuple[str, str]] = []
        for item in concepts:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            if not name:
                continue
            result.append((name, desc))
            if len(result) >= MAX_CONCEPTS:
                break
        return result

    @staticmethod
    async def extract_concepts(db_session: AsyncSession, kb_id: str, content: str) -> List[Tuple[str, str]]:
        model = llm_factory.create_model()
        sys_prompt = _CONCEPT_EXTRACT_SYS_PROMPT
        chat_result = await model.chat(
            sys_prompt,
            [{"role": "user", "content": content or ""}],
            {"temperature": 0.2},
        )
        raw = _get_chat_content(chat_result)
        return ConceptService.parse_extract_json(raw)

    @staticmethod
    def _parse_multi_merge_decision(raw: str) -> Tuple[List[ConceptMerge], List[str]]:
        obj = _safe_json_loads(raw)
        if not obj or not isinstance(obj, dict):
            return [], []

        merge = obj.get("merge", [])
        keep_ids = obj.get("keep_ids", [])
        if not isinstance(merge, list) or not isinstance(keep_ids, list):
            return [], []
        
        merge_list: List[ConceptMerge] = []
        for m in merge:
            if not isinstance(m, dict):
                continue
            source_ids = m.get("source_ids", [])
            name = m.get("name", "")
            description = m.get("description", "")
            if not name or not description:
                continue
            merge_list.append(ConceptMerge(
                source_ids=source_ids,
                name=name,
                description=description,
            ))
        
        return merge_list, keep_ids

    @staticmethod
    async def _decide_multi_merge(db_session: AsyncSession, kb_id: str, concepts: List[Dict[str, Any]]) -> Tuple[List[ConceptMerge], List[str]]:
        """
        让模型对多条相近概念做合并决策。
        concepts: [{"id":..., "name":..., "description":...}, ...]
        返回: (merge_list: List[ConceptMerge], keep_ids: List[str])
        """
        model = llm_factory.create_model()

        payload = json.dumps({"concepts": concepts}, ensure_ascii=False)
        chat_result = await model.chat(
            _CONCEPT_MULTI_MERGE_DECISION_SYS_PROMPT,
            [{"role": "user", "content": payload}],
            {"temperature": 0.2},
        )
        raw = _get_chat_content(chat_result)
        merge_list, keep_ids = ConceptService._parse_multi_merge_decision(raw)
        return merge_list, keep_ids

    @staticmethod
    async def _create_or_merge_one(
        db_session: AsyncSession,
        kb_id: str,
        doc_id: str,
        name: str,
        desc: Optional[str] = None,
    ) -> bool:
        name = (name or "").strip()
        if not name or len(name) > 256:
            return False
        desc = (desc or "").strip() or None

        candidates, _ = await ConceptService.get_by_kb_ids(
            db_session,
            [kb_id],
            keyword=name,
            limit=8,
        )

        payload_concepts: List[Dict[str, Any]] = [
            {"id": ConceptService._NEW_CONCEPT_TEMP_ID, "name": name, "description": desc or ""}
        ]
        payload_concepts.extend(
            [
                {"id": c.id, "name": c.concept_name, "description": c.description or ""}
                for c in candidates
            ]
        )

        if len(payload_concepts) <= 1:
            record = await ConceptService.add_or_update(db_session, kb_id, [doc_id], name, desc)
            if not record:
                return False
            return True

        merge_list, keep_ids = await ConceptService._decide_multi_merge(db_session, kb_id, payload_concepts)
        if not merge_list and not keep_ids:
            record = await ConceptService.add_or_update(db_session, kb_id, [doc_id], name, desc)
            if not record:
                return False
            return True
        
        for merge in merge_list:
            doc_ids: List[str] = []
            for source_id in merge.source_ids:
                # 是否对新概念进行了合并
                if source_id == ConceptService._NEW_CONCEPT_TEMP_ID:
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
                    continue
                # 如果 source_id 在 keep_ids 中，则不删除
                if source_id in keep_ids:
                    continue

                # 获取 source_id 对应的概念
                src = await ConceptService.get_by_kb_and_id(db_session, kb_id, source_id)
                if src:
                    for d in _norm_doc_ids(src.doc_ids):
                        if d not in doc_ids:
                            doc_ids.append(d)
                # 删除 source_id 对应的概念
                await ConceptService.delete_by_kb_and_id(db_session, kb_id, source_id)

            record = await ConceptService.add_or_update(db_session, kb_id, doc_ids, merge.name, merge.description)

        # 如果保持的新是新纪录，则直接插入数据
        for kid in keep_ids:
            if kid == ConceptService._NEW_CONCEPT_TEMP_ID:
                record = await ConceptService.add_or_update(db_session, kb_id, [doc_id], name, desc)
                if not record:
                    return False
                return True

        return True

    @staticmethod
    async def create_or_merge(
        db_session: AsyncSession,
        kb_id: str,
        doc_id: str,
        concepts: List[Tuple[str, str]],
    ) -> int:
        """
        批量写入或合并概念。若 (kb_id, concept_name) 已存在则合并描述并追加 doc_id；否则新增。
        concepts: [(概念名, 描述), ...]
        """
        if not concepts:
            return 0
        count = 0
        for name, desc in concepts:
            ok = await ConceptService._create_or_merge_one(db_session, kb_id, doc_id, name, desc)
            if ok:
                count += 1
        return count

    @staticmethod
    async def add_or_update(
        db_session: AsyncSession,
        kb_id: str,
        doc_ids: List[str],
        concept_name: str,
        description: Optional[str] = None,
    ) -> Optional[KBConcept]:
        name = (concept_name or "").strip()
        if not name or len(name) > 256:
            return None
        desc = (description or "").strip() or None

        record = await ConceptService.get_by_kb_and_name(db_session, kb_id, name)
        if record:
            source_doc_ids = _norm_doc_ids(record.doc_ids)
            for doc_id in doc_ids:
                if doc_id not in source_doc_ids:
                    source_doc_ids.append(doc_id)
            if desc: # 如果描述不为空，则更新描述
                record.description = desc
            record.doc_ids = source_doc_ids
            db_session.add(record)
            await db_session.commit()
            return record

        record = KBConcept(
            id=str(uuid.uuid4()),
            kb_id=kb_id,
            concept_name=name,
            description=desc,
            doc_ids=doc_ids,
        )
        db_session.add(record)
        await db_session.commit()
        return record

    @staticmethod
    async def get_by_kb_ids(
        db_session: AsyncSession,
        kb_ids: List[str],
        keyword: Optional[str] = None,
        limit: int = 200,
        page: Optional[int] = None,
        exclude_names: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None,
    ) -> Tuple[List[KBConcept], Optional[int]]:
        """
        按知识库 ID 列表查询概念。始终返回 (items, total)，不传 page 时 total 为 None，传 page 时 total 为总数。
        """
        kb_ids = [x for x in (kb_ids or []) if x]
        if not kb_ids:
            return [], (0 if page is not None else None)
        q = select(KBConcept).where(KBConcept.kb_id.in_(kb_ids))
        if keyword and keyword.strip():
            q = q.where(KBConcept.concept_name.ilike(f"%{keyword.strip()}%"))
        exclude_names = [x.strip() for x in (exclude_names or []) if x and x.strip()]
        if exclude_names:
            q = q.where(KBConcept.concept_name.notin_(exclude_names))
        exclude_ids = [x for x in (exclude_ids or []) if x]
        if exclude_ids:
            q = q.where(KBConcept.id.notin_(exclude_ids))
        q = q.order_by(KBConcept.updated_at.desc())

        if page is not None:
            total = (await db_session.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0
            q = q.offset((page - 1) * limit).limit(limit)
            result = await db_session.execute(q)
            return result.scalars().all(), total
            
        q = q.limit(limit)
        result = await db_session.execute(q)
        return result.scalars().all(), None

    @staticmethod
    async def get_by_kb_and_name(
        db_session: AsyncSession,
        kb_id: str,
        concept_name: str,
    ) -> Optional[KBConcept]:
        result = await db_session.execute(
            select(KBConcept).where(
                KBConcept.kb_id == kb_id,
                KBConcept.concept_name == (concept_name or "").strip(),
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_kb_and_id(
        db_session: AsyncSession,
        kb_id: str,
        concept_id: str,
    ) -> Optional[KBConcept]:
        result = await db_session.execute(
            select(KBConcept).where(
                KBConcept.kb_id == kb_id,
                KBConcept.id == concept_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def delete_by_kb_and_id(
        db_session: AsyncSession,
        kb_id: str,
        concept_id: str,
    ) -> int:
        if not concept_id:
            return 0
        result = await db_session.execute(
            delete(KBConcept).where(KBConcept.kb_id == kb_id, KBConcept.id == concept_id)
        )
        await db_session.commit()
        return result.rowcount or 0

    @staticmethod
    async def delete_by_kb_and_name(
        db_session: AsyncSession,
        kb_id: str,
        concept_name: str,
    ) -> int:
        name = (concept_name or "").strip()
        if not name:
            return 0
        result = await db_session.execute(
            delete(KBConcept).where(KBConcept.kb_id == kb_id, KBConcept.concept_name == name)
        )
        await db_session.commit()
        return result.rowcount or 0

    @staticmethod
    async def delete_all_by_kb(db_session: AsyncSession, kb_id: str) -> int:
        """删除知识库下所有概念，返回删除条数。"""
        result = await db_session.execute(delete(KBConcept).where(KBConcept.kb_id == kb_id))
        await db_session.commit()
        return result.rowcount or 0
    
    @staticmethod
    async def remove_doc(
        db_session: AsyncSession,
        kb_id: str,
        doc_id: str,
    ) -> int:
        """
        从概念中移除某文档关联：将 doc_id 从 doc_ids 中去掉；
        若某概念 doc_ids 变为空则删除该概念。返回被更新的概念数（含因清空而删除的）。
        """
        result = await db_session.execute(
            select(KBConcept).where(KBConcept.kb_id == kb_id)
        )
        rows = result.scalars().all()
        affected = 0
        for row in rows:
            doc_ids = _norm_doc_ids(row.doc_ids)
            if doc_id not in doc_ids:
                continue
            doc_ids = [x for x in doc_ids if x != doc_id]
            if not doc_ids:
                await db_session.delete(row)
                affected += 1
            else:
                row.doc_ids = doc_ids
                db_session.add(row)
                affected += 1
        await db_session.commit()
        return affected
