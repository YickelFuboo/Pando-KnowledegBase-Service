# 概念提取功能设计

## 一、目标

- 文档切片后，用模型从每个切片中识别「名词概念」，并写入概念表。
- 概念表按知识库维度组织，同一知识库下所有文档识别的概念都进入该库的概念表。
- 概念记录关联来源文档；文档删除时同步删除该文档产生的概念。
- 提供 API：按知识库查全部概念、按知识库 + 关键词模糊查询概念。

## 二、概念表设计（知识库粒度 + 关联文档）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | String(36) | 主键，UUID |
| kb_id | String(32) | 知识库ID，FK → knowledgebase.id |
| doc_id | String(36) | 文档ID，FK → documents.id，用于级联删除 |
| concept_name | String(256) | 概念名称 |
| created_at | DateTime | 创建时间 |

- **唯一约束**：不设 (kb_id, doc_id, concept_name) 唯一约束，允许多 chunk 重复提取同一概念；若需去重可在应用层做。
- **索引**：kb_id、doc_id、concept_name（便于按库查询、按文档删除、按名称模糊查）。

## 三、概念提取流程

- **时机**：在 `DocParserService._build_chunks` 得到 `docs` 之后，与 `_process_auto_keywords` / `_process_auto_questions` 同级，在写入向量库之前。
- **配置**：在 `parser_config` 中增加开关，如 `auto_concepts: true`（或数字表示最多提取数）。
- **逻辑**：对每个 chunk 的 `content_with_weight` 调用 LLM 提取名词概念；返回概念名列表；批量写入概念表，每条记录带 `kb_id`、`doc_id`、`concept_name`。
- **文档重新解析**：在 `delete_old` 为 True 时，在删除旧 chunks 的同时删除该文档的旧概念。

## 四、文档删除时同步删除概念

在 `DocumentService.delete_document_by_id` 中，在调用 `delete_document_chunks` 之后，调用删除该 `doc_id` 的概念记录。

## 五、概念 API 设计

| 接口 | 方法 | 说明 |
|------|------|------|
| 按知识库查概念 | GET /api/v1/knowledgebase/{kb_id}/concepts | 分页：page, page_size；返回该库下所有概念。 |
| 按知识库+关键词模糊查 | GET /api/v1/knowledgebase/{kb_id}/concepts?keyword=xxx | 对 concept_name 模糊匹配，分页同上。 |

返回结构：`{ "items": [ { "id", "concept_name", "doc_id", "document_name", "created_at" } ], "total", "page", "page_size" }`。
