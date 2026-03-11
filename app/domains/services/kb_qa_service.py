import logging
import time
import re
from copy import deepcopy
from functools import partial
from typing import Dict, Any, Generator, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.domains.services.kb_service import KBService
from app.domains.services.common.deep_research import DeepResearcher
from app.rag_core.search_api import RETRIEVALER, KG_RETRIEVALER
from app.rag_core.rag.prompts import kb_prompt, chunks_format, cross_languages, keyword_extraction, message_fit_in, full_question, citation_prompt
from app.rag_core.llm_service import LLMBundle, LLMType
from app.rag_core.rag.app.tag import label_question
from app.rag_core.utils import ParserType
from app.rag_core.rag.app.resume import forbidden_select_fields4resume
from app.infrastructure.web_search.tavily import Tavily
from app.rag_core.utils import num_tokens_from_string
from app.agent_frame.session.manager import session_manager
from app.agent_frame.session.message import Message
from app.domains.schemes.kb_qa import ChatMessage


KB_CHAT_PROMPT = """
角色：你是AI助手魔灵（Moling），请根据用户的问题，给出详细的回答。
任务：基于知识库信息总结并回答用户的问题。
要求和限制：
  - 不要编造信息，特别是数字。
  - 如果知识库中的信息与用户问题无关，请直接说：抱歉，没有找到相关信息。
  - 使用markdown格式回答。
  - 使用用户问题的语言回答。
  - 不要编造信息，特别是数字。

### 知识库信息
{knowledge}

以上是知识库中的信息。
"""


BAD_CITATION_PATTERNS = [
    re.compile(r"\(\s*ID\s*[: ]*\s*(\d+)\s*\)"),  # (ID: 12) - 圆括号格式
    re.compile(r"\[\s*ID\s*[: ]*\s*(\d+)\s*\]"),  # [ID: 12] - 方括号格式
    re.compile(r"【\s*ID\s*[: ]*\s*(\d+)\s*】"),  # 【ID: 12】 - 中文方括号格式
    re.compile(r"ref\s*(\d+)", flags=re.IGNORECASE),  # ref12、REF 12 - ref格式
]

DEFAULT_TOP_N = 12
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_VECTOR_SIMILARITY_WEIGHT = 0.3
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.1

class QAService:
    """问答服务类"""

    @staticmethod
    async def _decorate_answer(answer, kbinfos, prompt, embd_mdl, retriever, enable_quote):
        """装饰和格式化最终答案"""        
        # 分离思考过程和最终答案
        ans = answer.split("</think>")
        think = ""
        if len(ans) == 2:
            think = ans[0] + "</think>"
            answer = ans[1]

        refs = {}

        # 引用机制说明：prompt 中知识片段以 ID:0,1,2,... 表示（找回的 chunks[i] 的下标），答案中 [ID:i] 的 i 即该下标。
        # chunks对应的原始文档信息通过refs[]返回，refs[i] 与 chunks[i]  顺序一致
        # 所以，前台收到相应后如果想定位文档，可以通过answer中的 [ID:i]的下标，找到refs[i]，从而定位文档。
        if enable_quote:
            chunk_idxs = set([])
            
            # 如果答案中没有引用标记且启用了嵌入模型，自动插入引用
            if embd_mdl and not re.search(r"\[ID:([0-9]+)\]", answer):
                answer, chunk_idxs = await retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - DEFAULT_VECTOR_SIMILARITY_WEIGHT,
                    vtweight=DEFAULT_VECTOR_SIMILARITY_WEIGHT,
                )
            else:
                # 如果答案中已有引用标记，提取引用索引
                for match in re.finditer(r"\[ID:([0-9]+)\]", answer):
                    i = int(match.group(1))
                    if i < len(kbinfos["chunks"]):
                        chunk_idxs.add(i)

            # 修复错误的引用格式
            answer, chunk_idxs = QAService.repair_bad_citation_formats(answer, kbinfos, chunk_idxs)
            
            # 处理文档聚合信息，只保留被引用的文档
            doc_idxs = set([kbinfos["chunks"][int(i)]["doc_id"] for i in chunk_idxs])
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in doc_idxs]
            if not recall_docs:
                recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs  # 只保留被引用的文档
            
            # 准备引用信息，移除向量数据以减小响应大小
            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]
       
            # 转换原始内容引用部分字段名称
            refs["chunks"] = chunks_format(refs)
        
        # 处理API密钥错误
        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " 请在'用户设置 -> 模型提供商 -> API密钥'中设置LLM API密钥"
        
        # 返回结果
        return {"answer": think + answer, "reference": refs, "prompt": prompt, "created_at": time.time()}

    @staticmethod
    async def _pre_process_question(
        question: str,
        tenant_id: str,
        chat_mdl: LLMBundle,
        history_messages: Optional[List[ChatMessage]] = None,
        enable_multi_questions:Optional[bool] = True, 
        enable_keyword_extraction: Optional[bool] = False, 
        target_language: Optional[str] = None,
    ):
        try:
            # 如果有历史消息，则对多条消息进行总结（仅取到最近3次用户消息为止）
            if enable_multi_questions and len(history_messages) > 0:
                messages = []                
                # 先添加当前用户问题
                messages.append({"role": "user", "content": question})
                
                # 往前找历史消息，直到找到第3条用户消息为止
                user_count = 1  # 已经添加了当前问题
                for msg in reversed(history_messages):  # 从最新的历史消息开始往前找
                    messages.insert(0, msg.to_dict())  # 插入到列表开头
                    
                    # 如果遇到用户消息，计数+1
                    if msg.role == "user":
                        user_count += 1
                        # 如果已经找到第3条用户消息，停止
                        if user_count >= 3:
                            break
                
                # 对多条历史消息进行总结
                question = await full_question(tenant_id, LLMType.CHAT, messages)

            if enable_keyword_extraction:
                keyword = await keyword_extraction(chat_mdl, question)
                question += keyword

            if target_language:
                question = await cross_languages(tenant_id, LLMType.CHAT, question, [target_language])

            return question

        except Exception as e:
            logging.error(f"问题预处理问题失败: {e}")
            raise e

    @staticmethod
    async def kb_query(
        db: AsyncSession,          
        tenant_id: str,      # 租户ID
        user_id: str,        # 用户ID
        kb_ids: List[str],   # 知识库ID列表
        question: str,       # 用户问题
        session_id: Optional[str] = None,   # 会话ID，传入则从会话获取历史记录
        doc_ids: Optional[List[str]] = None, # 指定检索的文档ID列表
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        enable_quote: bool = True,
        enable_multi_questions: bool = True,        
        enable_keyword_extraction: bool = False,
        enable_deep_research: bool = False,        
        enable_web_search: bool = False,
        enable_knowledge_graph: bool = False,
        target_language: Optional[str] = None,
        is_stream: bool = False
        ):
        try:
            session = None
            active_session_id = session_id
            if active_session_id:
                session = await session_manager.get_session(active_session_id)

            if not session:
                active_session_id = await session_manager.create_session(
                    session_type="chat",
                    user_id=user_id,
                    description=question[:100] if question else "",
                    llm_name=(model_provider or "") + "-" + (model_name or "default"),
                )
                session = await session_manager.get_session(active_session_id)
                if not session:
                    raise RuntimeError("创建会话失败")

            if not session.description:
                session.description = question[:200] if question else ""

            # 获取历史消息
            history_messages: Optional[List[ChatMessage]] = None
            if session:
                history_messages = [
                    ChatMessage(role=m.role.value, content=m.content or "")
                    for m in session.get_history_for_context()
                ]

            await session_manager.add_message(active_session_id, Message.user_message(question))

            # 获取知识库信息
            kbs = await KBService.get_kb_by_ids(db, kb_ids)
            if not kbs:
                raise ValueError("知识库不存在")
            
            # 检查知识库是否使用相同的嵌入模型
            embedding_list = list(set([kb.embd_model_name for kb in kbs]))
            if len(embedding_list) > 1:
                raise ValueError("知识库使用了不同的嵌入模型，无法同时检索")
            
            # 创建模型
            embd_mdl = LLMBundle(kbs[0].tenant_id, LLMType.EMBEDDING, provider=kbs[0].embd_provider_name, model=kbs[0].embd_model_name)
            chat_mdl = LLMBundle(kbs[0].tenant_id, LLMType.CHAT, provider=model_provider, model=model_name)
            rerank_mdl = LLMBundle(kbs[0].tenant_id, LLMType.RERANK, provider=kbs[0].rerank_provider_name, model=kbs[0].rerank_model_name)
                        
            max_tokens = chat_mdl.max_length if hasattr(chat_mdl, 'max_length') else 8192
            tenant_ids = list(set([kb.tenant_id for kb in kbs]))
            kb_ids = [kb.id for kb in kbs]
            
            # 1. =====尝试使用SQL查询（如果知识库支持）
            field_map = await KBService.get_field_map(db, kb_ids)
            if field_map:
                # 禁用最新一条消息进行SQL查询
                logging.info(f" Try uuse SQL to retrieval: {question}")
                sql_result = await QAService.use_sql(
                    question, field_map, tenant_ids[0], chat_mdl, enable_quote
                )
                if sql_result:
                    answer_text = sql_result.get("answer", "")
                    await session_manager.add_message(active_session_id, Message.assistant_message(answer_text))
                    yield {**sql_result, "session_id": active_session_id}
                    return

            # 2. =====执行知识库检索
            # 处理用户问题
            question = await QAService._pre_process_question(
                question=question,
                tenant_id=tenant_id, 
                chat_mdl=chat_mdl,
                history_messages=history_messages,
                enable_multi_questions=enable_multi_questions, 
                enable_keyword_extraction=enable_keyword_extraction, 
                target_language=target_language,
            )

            # 初始化知识库信息
            thought = ""
            kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
            knowledges = []

            # 2.1 启动Deep Search方式检索
            if enable_deep_research:
                reasoner = DeepResearcher(
                    db,
                    chat_mdl,
                    {"tavily_api_key": settings.tavily_api_key, "use_kg": enable_knowledge_graph},
                    partial(RETRIEVALER.retrieval, 
                        embd_mdl=embd_mdl, 
                        tenant_ids=tenant_ids, 
                        kb_ids=kb_ids, 
                        page=1, 
                        page_size=DEFAULT_TOP_N, 
                        similarity_threshold=0.2, 
                        vector_similarity_weight=DEFAULT_VECTOR_SIMILARITY_WEIGHT
                    )
                )
                
                # 执行推理过程
                last_deep_answer = None
                async for think in reasoner.thinking(kbinfos, question):
                    if isinstance(think, str):
                        # 如果是字符串，保存思考过程和知识
                        thought = think
                        knowledges = [t for t in think.split("\n") if t]
                    elif is_stream:
                        # 如果是流式输出，直接返回，统一带上 session_id
                        last_deep_answer = think.get("answer", think) if isinstance(think, dict) else think
                        if isinstance(think, dict):
                            yield {**think, "session_id": active_session_id}
                        else:
                            yield {"answer": think, "session_id": active_session_id}
                if last_deep_answer is not None:
                    answer_content = last_deep_answer.get("answer", last_deep_answer) if isinstance(last_deep_answer, dict) else last_deep_answer
                    await session_manager.add_message(active_session_id, Message.assistant_message(str(answer_content)))
            else:     
                # 2.2 执行知识库检索
                if embd_mdl:
                    kbinfos = await RETRIEVALER.retrieval(
                        question=question,
                        embd_mdl=embd_mdl,
                        tenant_ids=tenant_ids,
                        kb_ids=kb_ids,
                        page=1,
                        page_size=DEFAULT_TOP_N,
                        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                        vector_similarity_weight=DEFAULT_VECTOR_SIMILARITY_WEIGHT,
                        doc_ids=doc_ids if doc_ids else None,
                        top=DEFAULT_TOP_K,
                        aggs=False,
                        rank_feature=await label_question(db, question, kbs)
                    )

                # 2.3 集成Tavily外部知识源
                if enable_web_search:
                    try:
                        tav = Tavily()
                        tav_res = await tav.retrieve_chunks(question)
                        if tav_res and tav_res.get("chunks"):
                            kbinfos["chunks"].extend(tav_res["chunks"])
                            kbinfos["total"] = len(kbinfos["chunks"])
                        if tav_res and tav_res.get("doc_aggs"):
                            kbinfos["doc_aggs"].extend(tav_res["doc_aggs"])
                    except Exception as e:
                        logging.warning(f"Tavily外部知识源检索失败: {e}")
                
                # 2.4 集成知识图谱检索
                if enable_knowledge_graph:
                    try:
                        ck = await KG_RETRIEVALER.retrieval(
                            question, 
                            tenant_ids, 
                            kb_ids, 
                            embd_mdl, 
                            chat_mdl
                        )
                        if ck and ck.get("content_with_weight"):
                            kbinfos["chunks"].insert(0, ck)  # 将知识图谱结果插入到最前面
                    except Exception as e:
                        logging.warning(f"知识图谱检索失败: {e}")
                            
                # 格式化知识库内容
                knowledges = await kb_prompt(db, kbinfos, max_tokens)

            if not knowledges:
                yield {
                    "answer": "抱歉，没有找到相关信息。",
                    "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                    "prompt": "",
                    "created_at": time.time(),
                    "session_id": active_session_id
                }
                return
            
            # 构建系统提示次消息列表
            system_prompt = KB_CHAT_PROMPT.format(knowledge="\n------\n" + "\n\n------\n\n".join(knowledges))
            if enable_quote:
                system_prompt += citation_prompt()

            # 构建消息清单，包含最新用户问题和历史消息，不包含System消息
            msgs = []
            if history_messages:
                for m in history_messages:
                    if m.role != "system":
                        content = re.sub(r"##\d+\$\$", "", m.content)
                        msgs.append({"role": m.role, "content": content})
            # 添加用户问题
            msgs.append({"role": "user", "content": question})
            
            # 生成回答
            if is_stream:
                # 流式生成：chat_stream 每次 yield 的是当次的一小段 delta，直接累加后推送
                answer = ""
                async for ans in chat_mdl.chat_stream(system_prompt, msgs, {"temperature": DEFAULT_TEMPERATURE}):
                    if thought:
                        ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
                    if not ans:
                        continue
                    answer += ans
                    yield {"answer": thought + answer, "reference": {}, "session_id": active_session_id}

                # 返回最终装饰后的完整答案
                final_answer = thought + answer
                last = await QAService._decorate_answer(final_answer, kbinfos, system_prompt, embd_mdl, RETRIEVALER, enable_quote)
                if not last.get("answer") and kbinfos.get("chunks"):
                    last["answer"] = "抱歉，模型未能生成回答，请重试。"
                # 只把最终答案压紧History，不压reference
                await session_manager.add_message(active_session_id, Message.assistant_message(last.get("answer") or final_answer))
                yield {**last, "session_id": active_session_id}
            else:
                # 非流式输出模式：一次性生成完整答案
                answer = await chat_mdl.chat(system_prompt, msgs, {"temperature": DEFAULT_TEMPERATURE})

                # 装饰答案
                result = await QAService._decorate_answer(answer, kbinfos, system_prompt, embd_mdl, RETRIEVALER, enable_quote)
                await session_manager.add_message(active_session_id, Message.assistant_message(result.get("answer") or answer))  #只把最终答案压紧History，不压reference
                yield {**result, "session_id": active_session_id}

        except Exception as e:
            logging.error(f"多轮对话服务执行失败: {e}")
            error_response = {
                "answer": f"抱歉，处理您的问题时出现了错误：{str(e)}",
                "reference": {"total": 0, "chunks": [], "doc_aggs": []},
                "prompt": "",
                "created_at": time.time(),
                "session_id": active_session_id
            }
            yield error_response


    @staticmethod
    async def use_sql(question: str, field_map: dict, tenant_id: str, chat_mdl, quota: bool = True):
        """
        使用SQL查询结构化数据
        
        此函数使用LLM将自然语言问题转换为SQL查询，然后执行查询并返回格式化的结果。
        主要用于处理结构化数据（如简历、表格等）的查询。
        
        Args:
            question (str): 用户的自然语言问题
            field_map (dict): 字段映射，描述数据库表结构
            tenant_id (str): 租户ID
            chat_mdl: 聊天模型实例，用于生成SQL
            quota (bool): 是否启用引用功能，默认True
            
        Returns:
            dict: 包含answer、reference、prompt字段的响应字典，如果失败则返回None
        """
        # 构建系统提示词
        sys_prompt = "You are a Database Administrator. You need to check the fields of the following tables based on the user's list of questions and write the SQL corresponding to the last question."
        
        # 构建用户提示词，包含表结构和问题
        user_prompt = f"""
Table name: {tenant_id};
Table of database fields are as follows:
{chr(10).join([f"{k}: {v}" for k, v in field_map.items()])}

Question are as follows:
{question}
Please write the SQL, only SQL, without any other explanations or text.
"""
        tried_times = 0

        async def get_table():
            """
            获取SQL查询结果
            
            使用LLM生成SQL，执行查询并返回结果。
            包含SQL清理和优化逻辑。
            
            Returns:
                tuple: (查询结果表, SQL语句)
            """
            nonlocal sys_prompt, user_prompt, question, tried_times
            
            # 使用LLM生成SQL
            sql = await chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.06})
            
            # 清理LLM输出，移除思考过程
            sql = re.sub(r"^.*</think>", "", sql, flags=re.DOTALL)
            logging.debug(f"{question} ==> {user_prompt} get SQL: {sql}")
            
            # SQL清理和标准化
            sql = re.sub(r"[\r\n]+", " ", sql.lower())  # 移除换行符
            sql = re.sub(r".*select ", "select ", sql.lower())  # 确保以SELECT开头
            sql = re.sub(r" +", " ", sql)  # 标准化空格
            sql = re.sub(r"([;；]|```).*", "", sql)  # 移除分号后的内容
            
            # 验证SQL格式
            if sql[: len("select ")] != "select ":
                return None, None
                
            # 优化SQL查询，确保包含必要的字段
            if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
                if sql[: len("select *")] != "select *":
                    # 如果不是SELECT *，添加doc_id和docnm_kwd字段
                    sql = "select doc_id,docnm_kwd," + sql[6:]
                else:
                    # 如果是SELECT *，替换为具体字段列表
                    flds = []
                    for k in field_map.keys():
                        if k in forbidden_select_fields4resume:
                            continue
                        if len(flds) > 11:  # 限制字段数量
                            break
                        flds.append(k)
                    sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

            logging.debug(f"{question} get SQL(refined): {sql}")
            tried_times += 1
            
            # 执行SQL查询
            return RETRIEVALER.sql_retrieval(sql, format="json"), sql

        tbl, sql = await get_table()
        if tbl is None:
            return None
        if tbl.get("error") and tried_times <= 2:
            user_prompt = f"""
            Table name: {tenant_id};
            Table of database fields are as follows:
            {chr(10).join([f"{k}: {v}" for k, v in field_map.items()])}

            Question are as follows:
            {question}
            Please write the SQL, only SQL, without any other explanations or text.


            The SQL error you provided last time is as follows:
            {sql}

            Error issued by database as follows:
            {tbl["error"]}

            Please correct the error and write SQL again, only SQL, without any other explanations or text.
            """
            tbl, sql = await get_table()
            logging.debug("TRY it again: {}".format(sql))

        logging.debug("GET table: {}".format(tbl))
        if tbl.get("error") or len(tbl["rows"]) == 0:
            return None

        docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "doc_id"])
        doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "docnm_kwd"])
        column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

        # compose Markdown table
        columns = (
            "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"], tbl["columns"][i]["name"])) for i in column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")
        )

        line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

        rows = ["|" + "|".join([str(r[i]).replace("None", " ") for i in column_idx]).replace("None", " ") + "|" for r in tbl["rows"]]
        rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
        if quota:
            rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
        else:
            rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
        rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

        if not docid_idx or not doc_name_idx:
            logging.warning("SQL missing field: " + sql)
            return {"answer": "\n".join([columns, line, rows]), "reference": {}, "prompt": sys_prompt, "created_at": time.time()}

        docid_idx = list(docid_idx)[0]
        doc_name_idx = list(doc_name_idx)[0]
        doc_aggs = {}
        for r in tbl["rows"]:
            if r[docid_idx] not in doc_aggs:
                doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
            doc_aggs[r[docid_idx]]["count"] += 1
        return {
            "answer": "\n".join([columns, line, rows]),
            "reference": {
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "total": len(tbl["rows"]),
                "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
                "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
            },
            "prompt": sys_prompt,
        }

    @staticmethod
    def repair_bad_citation_formats(answer: str, kbinfos: dict, idx: set):
        """
        修复错误的引用格式
        
        将LLM生成的各种不标准引用格式统一转换为标准的[ID:X]格式。
        这确保了引用的一致性和可解析性。
        
        Args:
            answer (str): 包含引用的答案文本
            kbinfos (dict): 知识库信息，包含chunks列表
            idx (set): 已识别的引用索引集合，会被更新
            
        Returns:
            tuple: (修复后的答案文本, 更新后的引用索引集合)
            
        Example:
            转换前: "根据文档(ID: 1)和[ID: 2]的内容，参考文献【ID: 3】、引用ref4显示..."
            转换后: "根据文档[ID:1]和[ID:2]的内容，参考文献[ID:3]、引用[ID:4]显示..."
        """
        max_index = len(kbinfos["chunks"])

        def safe_add(i):
            """
            安全地添加引用索引
            
            检查索引是否在有效范围内，如果是则添加到索引集合中。
            
            Args:
                i (int): 要添加的索引
                
            Returns:
                bool: 是否成功添加
            """
            if 0 <= i < max_index:
                idx.add(i)
                return True
            return False

        def find_and_replace(pattern, group_index=1, repl=lambda i: f"ID:{i}", flags=0):
            """
            查找并替换匹配的引用格式
            
            Args:
                pattern: 正则表达式模式
                group_index: 捕获组索引
                repl: 替换函数
                flags: 正则表达式标志
            """
            nonlocal answer

            def replacement(match):
                """
                替换匹配的引用格式为标准格式
                
                Args:
                    match: 正则表达式匹配对象
                    
                Returns:
                    str: 替换后的文本
                """
                try:
                    i = int(match.group(group_index))
                    if safe_add(i):
                        return f"[{repl(i)}]"
                except Exception:
                    pass
                return match.group(0)

            answer = re.sub(pattern, replacement, answer, flags=flags)

        # 遍历所有错误格式模式并进行修复
        for pattern in BAD_CITATION_PATTERNS:
            find_and_replace(pattern)

        return answer, idx