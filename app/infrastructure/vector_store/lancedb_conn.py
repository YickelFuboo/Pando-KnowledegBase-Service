import asyncio
import logging
from typing import Any, Optional

import lancedb
import numpy as np
from pandas import DataFrame

from .base import FusionExpr, MatchDenseExpr, MatchTextExpr, SearchRequest, SortOrder, VectorStoreConnection


class LanceDBConnection(VectorStoreConnection):
    """LanceDB连接实现。"""

    def __init__(self, uri: str):
        self.uri = uri
        self.db = lancedb.connect(uri)

    def get_db_type(self) -> str:
        return "lancedb"

    async def close(self):
        return

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self.db.table_names)
            return True
        except Exception as e:
            logging.error(f"LanceDB健康检查失败: {e}")
            return False

    async def create_space(self, space_name: str, vector_size: int, **kwargs) -> bool:
        try:
            if await self.space_exists(space_name):
                return True
            return True
        except Exception as e:
            logging.error(f"创建LanceDB空间失败 {space_name}: {e}")
            return False

    async def delete_space(self, space_name: str, **kwargs) -> bool:
        try:
            exists = await self.space_exists(space_name)
            if not exists:
                return True
            await asyncio.to_thread(self.db.drop_table, space_name)
            return True
        except Exception as e:
            logging.error(f"删除LanceDB空间失败 {space_name}: {e}")
            return False

    async def space_exists(self, space_name: str, **kwargs) -> bool:
        table_names = await asyncio.to_thread(self.db.table_names)
        return space_name in table_names

    async def insert_records(self, space_name: str, records: list[dict[str, Any]], **kwargs) -> list[str]:
        if not records:
            return []
        try:
            exists = await self.space_exists(space_name)
            if not exists:
                await asyncio.to_thread(self.db.create_table, space_name, data=records)
                return []
            table = await self._open_table(space_name)
            await asyncio.to_thread(table.add, records)
            return []
        except Exception as e:
            logging.error(f"LanceDB插入失败 {space_name}: {e}")
            return [str(e)]

    async def update_records(self, space_name: str, condition: dict[str, Any], new_value: dict[str, Any], fields_to_remove: list[str] = None, **kwargs) -> bool:
        try:
            records = await self._read_records(space_name)
            if not records:
                return True
            fields_to_remove = fields_to_remove or []
            updated = []
            changed = False
            for item in records:
                if self._match_condition(item, condition):
                    changed = True
                    for field_name in fields_to_remove:
                        item.pop(field_name, None)
                    for key, value in new_value.items():
                        if key == "id":
                            continue
                        item[key] = value
                updated.append(item)
            if not changed:
                return True
            return await self._rewrite_table(space_name, updated)
        except Exception as e:
            logging.error(f"LanceDB更新失败 {space_name}: {e}")
            return False

    async def delete_records(self, space_name: str, condition: dict[str, Any], **kwargs) -> int:
        try:
            records = await self._read_records(space_name)
            if not records:
                return 0
            kept = []
            deleted = 0
            for item in records:
                if self._match_condition(item, condition):
                    deleted += 1
                else:
                    kept.append(item)
            await self._rewrite_table(space_name, kept)
            return deleted
        except Exception as e:
            logging.error(f"LanceDB删除失败 {space_name}: {e}")
            return 0

    async def get_record(self, space_names: list[str], record_id: str, **kwargs) -> Optional[dict[str, Any]]:
        for space_name in space_names:
            records = await self._read_records(space_name)
            for item in records:
                if str(item.get("id")) == str(record_id):
                    return item
        return None

    async def search(self, space_names: list[str], request: SearchRequest, **kwargs) -> dict[str, Any]:
        all_docs = []
        for space_name in space_names:
            all_docs.extend(await self._read_records(space_name))
        if not all_docs:
            return {"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}}

        filtered = [doc for doc in all_docs if self._match_condition(doc, request.condition or {})]
        scored = [{"doc": doc, "score": 0.0} for doc in filtered]

        if request.match_exprs:
            for match_expr in request.match_exprs:
                if isinstance(match_expr, MatchTextExpr):
                    scored = self._apply_text_match(scored, match_expr)
                elif isinstance(match_expr, MatchDenseExpr):
                    scored = self._apply_dense_match(scored, match_expr)
                elif isinstance(match_expr, FusionExpr):
                    continue

        if request.order_by:
            for sort_field in reversed(request.order_by):
                reverse = sort_field.sort_order == SortOrder.DESC or sort_field.sort_order == SortOrder.DESC.value
                scored.sort(key=lambda x: x["doc"].get(sort_field.sort_field), reverse=reverse)
        else:
            scored.sort(key=lambda x: x["score"], reverse=True)

        total = len(scored)
        start = max(request.offset, 0)
        end = start + request.limit if request.limit > 0 else None
        page_docs = scored[start:end]

        hits = []
        for item in page_docs:
            doc = dict(item["doc"])
            if request.select_fields:
                selected = {k: doc.get(k) for k in request.select_fields if k in doc}
                if "id" in doc:
                    selected["id"] = doc["id"]
                doc = selected
            hits.append({"_id": str(doc.get("id", "")), "_score": float(item["score"]), "_source": doc})

        aggs = {}
        if request.agg_fields:
            for field_name in request.agg_fields:
                buckets = {}
                for item in scored:
                    value = item["doc"].get(field_name)
                    if value is None:
                        continue
                    if isinstance(value, list):
                        for v in value:
                            buckets[v] = buckets.get(v, 0) + 1
                    else:
                        buckets[value] = buckets.get(value, 0) + 1
                aggs[f"aggs_{field_name}"] = {
                    "buckets": [{"key": k, "doc_count": v} for k, v in buckets.items()]
                }

        return {"hits": {"total": {"value": total}, "hits": hits}, "aggregations": aggs}

    def get_total(self, result) -> int:
        try:
            return int(result.get("hits", {}).get("total", {}).get("value", 0))
        except Exception:
            return 0

    def get_chunk_ids(self, result) -> list[str]:
        try:
            return [str(hit.get("_id", "")) for hit in result.get("hits", {}).get("hits", [])]
        except Exception:
            return []

    def get_source(self, result) -> list[dict[str, Any]]:
        sources = []
        for hit in result.get("hits", {}).get("hits", []):
            source = self._to_python(dict(hit.get("_source", {})))
            source["id"] = hit.get("_id")
            source["_score"] = hit.get("_score", 0.0)
            sources.append(source)
        return sources

    def get_fields(self, result, fields: list[str]) -> dict[str, dict]:
        data = {}
        for source in self.get_source(result):
            sid = str(source.get("id", ""))
            if not sid:
                continue
            payload = {}
            for field_name in fields:
                if field_name in source and source[field_name] is not None:
                    payload[field_name] = source[field_name]
            if payload:
                data[sid] = payload
        return data

    def get_highlight(self, result, keywords: list[str], field_name: str) -> dict[str, Any]:
        return {}

    def get_aggregation(self, result, field_name: str) -> dict[str, Any]:
        return result.get("aggregations", {}).get(f"aggs_{field_name}", {})

    async def sql(self, sql: str, fetch_size: int, format: str, *args, **kwargs):
        raise NotImplementedError("LanceDB暂不支持SQL接口")

    async def _open_table(self, space_name: str):
        return await asyncio.to_thread(self.db.open_table, space_name)

    async def _read_records(self, space_name: str) -> list[dict[str, Any]]:
        if not await self.space_exists(space_name):
            return []
        table = await self._open_table(space_name)
        frame = await asyncio.to_thread(table.to_pandas)
        if not isinstance(frame, DataFrame) or frame.empty:
            return []
        records = frame.to_dict(orient="records")
        normalized = []
        for item in records:
            normalized.append(self._normalize_record(item))
        return normalized

    async def _rewrite_table(self, space_name: str, records: list[dict[str, Any]]) -> bool:
        try:
            if not records:
                await self.delete_space(space_name)
                return True
            await asyncio.to_thread(self.db.create_table, space_name, data=records, mode="overwrite")
            return True
        except Exception as e:
            logging.error(f"LanceDB重写表失败 {space_name}: {e}")
            return False

    def _normalize_record(self, item: dict[str, Any]) -> dict[str, Any]:
        return self._to_python(item)

    def _to_python(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [self._to_python(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: self._to_python(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_python(v) for v in value]
        return value

    def _match_condition(self, item: dict[str, Any], condition: dict[str, Any]) -> bool:
        if not condition:
            return True
        for key, value in condition.items():
            if key == "exists":
                if not item.get(value):
                    return False
                continue
            if isinstance(value, list):
                if item.get(key) not in value:
                    return False
            else:
                if item.get(key) != value:
                    return False
        return True

    def _apply_text_match(self, scored: list[dict[str, Any]], expr: MatchTextExpr) -> list[dict[str, Any]]:
        target = (expr.matching_text or "").lower()
        if not target:
            return scored
        result = []
        for item in scored:
            doc = item["doc"]
            score_boost = 0.0
            for field_name in expr.fields:
                raw = doc.get(field_name)
                if raw is None:
                    continue
                text = str(raw).lower()
                if target in text:
                    score_boost += 1.0
            if score_boost > 0:
                result.append({"doc": doc, "score": item["score"] + score_boost})
        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:expr.topn] if expr.topn > 0 else result

    def _apply_dense_match(self, scored: list[dict[str, Any]], expr: MatchDenseExpr) -> list[dict[str, Any]]:
        query_vec = np.asarray(expr.embedding_data, dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return scored
        similarity = float(expr.extra_options.get("similarity", 0.0))
        result = []
        for item in scored:
            doc = item["doc"]
            vec = doc.get(expr.vector_column_name)
            if vec is None:
                continue
            vec_arr = np.asarray(vec, dtype=float)
            denom = np.linalg.norm(vec_arr) * query_norm
            if denom == 0:
                continue
            cos_score = float(np.dot(query_vec, vec_arr) / denom)
            if cos_score >= similarity:
                result.append({"doc": doc, "score": cos_score})
        result.sort(key=lambda x: x["score"], reverse=True)
        return result[:expr.topn] if expr.topn > 0 else result
