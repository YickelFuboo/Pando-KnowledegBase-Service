import re
import json
import logging
import copy
import random
import xxhash
import asyncio
import numpy as np
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from timeit import default_timer as timer
from app.domains.models import Document, KB, ProcessStatus
from app.infrastructure.database import get_db
from app.infrastructure.llms import llm_factory, embedding_factory
from app.infrastructure.storage import STORAGE_CONN
from app.infrastructure.vector_store import VECTOR_STORE_CONN
from app.utils.progress_callback import ProgressCallback
from app.domains.services.common.doc_vector_store_service import DocVectorStoreService
from app.domains.services.common.file_service import FileService, FileUsage
from app.domains.services.doc_service import DocumentService
from app.rag_core.constants import CHAT_LIMITER, CHUNK_LIMITER, MINIO_LIMITER, KG_LIMITER, TAG_FLD, PAGERANK_FLD, DOC_BULK_SIZE, EMBEDDING_BATCH_SIZE
from app.rag_core.utils import ParserType, truncate, num_tokens_from_string
from app.rag_core.chunk_api import CHUNK_FACTORY
from app.rag_core.deepdoc.parser import PdfParser, ExcelParser
from app.rag_core.rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from app.rag_core.rag.nlp import rag_tokenizer
from app.rag_core.rag.prompts import keyword_extraction, question_proposal, content_tagging
from app.rag_core.graphrag.utils import get_llm_cache, set_llm_cache, get_tags_from_cache, set_tags_to_cache
from app.rag_core.graphrag.general.index import run_graphrag
from app.rag_core.search_api import RETRIEVALER
from app.rag_core.llm_service import LLMBundle, LLMType


class DocParserService:
    """文档解析服务类"""

    def __init__(self, session: AsyncSession, kb: KB, document: Document, user_id: str, delete_old: bool = True):
        self.kb = kb
        self.document = document
        self.user_id = user_id
        self.db_session = session
        self.delete_old = delete_old

        self.parser_config = json.loads(document.parser_config) if isinstance(document.parser_config, str) else document.parser_config

        # 创建模型实例
        self.chat_model = LLMBundle(self.kb.tenant_id, LLMType.CHAT)
        self.embedding_model = LLMBundle(self.kb.tenant_id, LLMType.EMBEDDING, provider=self.kb.embd_provider_name, model=self.kb.embd_model_name)

        # 创建向量存储服务
        self.vector_store = DocVectorStoreService(VECTOR_STORE_CONN)
        # 文件存储服务
        self.file_store = STORAGE_CONN

        # 创建回调处理对象
        self.callback = ProgressCallback()

    async def parse_document(self) -> bool:
        """解析文档内容（异步任务）"""
        try:
            if not self.document:
                raise ValueError("文档不存在")
            
            # 清空旧的数据
            if self.delete_old:
                await DocumentService.delete_document_chunks(self.db_session, self.document.id)

            # 获取文件内容
            file_content = await FileService.get_file_content(
                self.document.file_id,
                FileUsage.DOCUMENT
            )
            
            await DocumentService.update_document_status(self.db_session, self.document.id, ProcessStatus.CHUNKING)
            # 把文档按照页码拆解分为多个子任务
            sub_tasks = await self._create_parser_tasks(file_content) 
            # 多线程执行子任务
            for task in sub_tasks:
                await self._execute_parser_task(file_content, task)

            # 执行RAPTOR任务
            await DocumentService.update_document_status(self.db_session, self.document.id, ProcessStatus.RAPTORING)
            if self.parser_config.get("raptor", {}).get("use_raptor", False):
                await self._execute_raptor_task()

            # 执行GraphRAG任务
            await DocumentService.update_document_status(self.db_session, self.document.id, ProcessStatus.GRAPHING)
            if self.parser_config.get("graphrag", {}).get("use_graphrag", False):
                await self._execute_graphrag_task()

            return True            
        except Exception as e:
            logging.error(f"文档解析失败: {self.document.id}, 错误: {e}")
            raise

    async def _create_parser_tasks(self,file_content: bytes) -> List[Dict[str, Any]]:
        """
        根据配置，按照页码把文档解析任务拆分成多个子任务
        
        Args:
            file_content: 文件内容
            
        Returns:
            List[Dict[str, Any]]: 子任务列表，每个任务包含task_params
        """
        try:            
            # 按照文档页创建分批任务
            sub_tasks = []
            
            if self.document.type == "pdf":
                # PDF文档：按页范围分片
                do_layout = self.parser_config.get("layout_recognize", "DeepDOC")
                    
                page_size = self.parser_config.get("task_page_size") or 12
                if self.document.parser_id == "paper":
                    page_size = self.parser_config.get("task_page_size") or 22
                if self.document.parser_id in ["one", "knowledge_graph"] or do_layout != "DeepDOC":
                    page_size = 10 ** 9
                    
                page_ranges = self.parser_config.get("pages") or [(1, 10 ** 5)]

                pages = PdfParser.total_page_number(self.document.name, file_content)
                if pages is None:
                    pages = 0

                # 创建分页任务
                for start_page, end_page in page_ranges:
                    start_page = max(0, start_page - 1)  # 转换为从0开始
                    end_page = min(end_page - 1, pages)
                    for p in range(start_page, end_page, page_size):
                        task = {
                            "from_page": p,
                            "to_page": min(p + page_size, end_page)
                        }
                        sub_tasks.append(task)
                        
            elif self.document.parser_id == "table":
                # Excel表格：按行分片
                total_rows = ExcelParser.row_number(self.document.name, file_content)
                if total_rows is None:
                    total_rows = 0
                    
                row_batch_size = 3000                
                for i in range(0, total_rows, row_batch_size):
                    task = {
                        "from_page": i,
                        "to_page": min(i + row_batch_size, total_rows)
                    }
                    sub_tasks.append(task)
            else:
                # 其他类型：单个任务
                task = {
                    "from_page": 0,
                    "to_page": 10 ** 5
                }
                sub_tasks.append(task)

            # 原是项目这里有，寻找历史task，如果任务内容完全一样，直接服用历史任务结果的chunks
            # 原有项目，这里吧每个子任务的状态加入任务队列表。
            
            logging.info(f"文档 {self.document.id} 创建了 {len(sub_tasks)} 个子任务")
            return sub_tasks
            
        except Exception as e:
            logging.error(f"创建子任务失败: {e}")
            raise

    async def _execute_parser_task(self, file_content: bytes, task: Dict[str, Any]):
        """
        执行单个子解析任务
        
        Args:
            file_content: 文件内容
            task: 任务配置
            
        Returns:
            Dict[str, Any]: 执行结果，包含解析内容、切片、向量等信息
        """
        
        logging.info(f"开始执行子解析任务: 文档 {self.document.id}, 页数范围: {task['from_page']}-{task['to_page']}")
        start_ts = timer()
        try: 
            # 创建向量存储空间
            vector_size = await self.embedding_model.get_embedding_vector_size()
            await self.vector_store.createIdx(self.kb.tenant_id, self.kb.id, vector_size)

            # 内容切片
            chunk_start_ts = timer()
            chunks = await self._build_chunks(file_content, task)
            logging.info(f"文档 {self.document.id} 分片完成，耗时 {timer() - chunk_start_ts:.2f}秒")
            if not chunks:
                logging.error(f"文档分片未生成切片: {self.document.name}, 页数范围: {task['from_page']}-{task['to_page']}")
                return

            # chunk向量化
            embed_start_ts = timer()
            token_count, vector_size = await self._embedding_chunks(chunks)
            logging.info("Embedding chunks ({:.2f}s)".format(timer() - embed_start_ts))

            # 存储向量结果
            store_start_ts = timer()
            success = await self._store_chunks_vector(chunks)
            if not success:
                raise Exception("存储向量结果失败")
            logging.info("Store chunks vector ({:.2f}s)".format(timer() - store_start_ts))

        except Exception as e:
            logging.error(f"执行子解析任务失败: {e}")
            raise

        logging.info(f"Chunk doc({self.document.name}), page({task['from_page']}-{task['to_page']}), chunks({len(chunks)}), token({token_count}), elapsed:{timer() - start_ts:.2f}")

    async def _build_chunks(self, file_content: bytes, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        构建切片
        """
        try:            
            chunker = CHUNK_FACTORY[self.document.parser_id.lower()]
            
            if not chunker:
                raise ValueError(f"未找到对应的Chunker: {self.document.parser_id}")
            
            if not hasattr(chunker, 'chunk'):
                raise ValueError(f"Chunker {chunker} 没有chunk方法")
            
            async with CHUNK_LIMITER:
                chunks = await chunker.chunk(
                    self.document.name, 
                    binary=file_content, 
                    from_page=task["from_page"],
                    to_page=task["to_page"], 
                    lang=self.kb.language,
                    callback=self.callback.progress_callback,
                    tenant_id=self.kb.tenant_id,
                    kb_id=self.document.kb_id,
                    parser_config=self.parser_config
                )

            # 处理每个切片，chunk转换为doc，（1）新增doc_id、kb_id信息  （2）保存chunk中图片信息到文件存储
            docs = await self._process_chunks_and_store_image(chunks)

            # 自动提取关键词
            if self.parser_config.get("auto_keywords", 0):
                docs = await self._process_auto_keywords(docs)
                
            # 自动提取Question    
            if self.parser_config.get("auto_questions", 0):
                docs = await self._process_auto_questions(docs)

            # 处理标签
            if self.parser_config.get("tag_kb_ids", []):
                docs = await self._process_auto_tags(docs)

            # 概念提取
            if self.parser_config.get("auto_concepts", False):
                docs = await self._process_concept_extraction(docs)

            return docs
            
        except Exception as e:
            logging.error(f"构建切片失败: 文档 {self.document.id}, 错误: {e}")
            raise

    async def _process_chunks_and_store_image(self, chunks):
        """
        处理切片：
        1. 新增doc_id、kb_id信息
        2. 保存chunk中图片信息到文件存储
        
        Args:
            chunks: 切片数据，包含内容和可能的图片
        """
        docs = []       
        try:         
            for cur_chunk in chunks:
                doc = {
                    "doc_id": self.document.id,
                    "kb_id": str(self.document.kb_id)
                }
                if self.kb.page_rank > 0:
                    doc[PAGERANK_FLD] = self.kb.page_rank

                doc.update(cur_chunk)
                doc["id"] = xxhash.xxh64((cur_chunk["content_with_weight"] + str(doc["doc_id"])).encode("utf-8")).hexdigest()
                doc["create_time"] = str(datetime.now()).replace("T", " ")[:19]
                doc["create_timestamp_flt"] = datetime.now().timestamp()
                
                # 图片"image"信息来自与chunk
                # 如果没有图片，直接添加到结果列表
                if not doc.get("image"):
                    _ = doc.pop("image", None)
                    doc["img_id"] = ""
                    docs.append(doc)
                    continue

                # 如果chunk中有图片，则把图片上传，并把上传结果地址存储存储doc中
                output_buffer = BytesIO()
                try:
                    if isinstance(doc["image"], bytes):
                        output_buffer.write(doc["image"])
                        output_buffer.seek(0)
                    else:
                        # 如果图片是RGBA模式，转换为RGB模式后保存为JPEG格式
                        if doc["image"].mode in ("RGBA", "P"):
                            converted_image = doc["image"].convert("RGB")
                            # doc["image"].close()  存在多个chunk共享图片对象，所以使用延迟释放，在最后统一清理
                            doc["orig_image"] = doc["image"]  # 保存原始对象，用于后续延迟释放
                            doc["image"] = converted_image
                        doc["image"].save(output_buffer, format='JPEG')

                    # 上传到存储服务
                    if self.file_store:
                        async with MINIO_LIMITER:
                            # 将bytes对象包装成BytesIO对象
                            file_data_io = BytesIO(output_buffer.getvalue())
                            await self.file_store.put(
                                file_index=doc["id"], 
                                file_data=file_data_io, 
                                bucket_name=self.kb.id,
                            )
                        doc["img_id"] = "{}-{}".format(self.kb.id, doc["id"])    

                    docs.append(doc)                
                finally:
                    output_buffer.close()  # 确保BytesIO总是被关闭

            # 清理图片引用，最终doc中图片只保存存储信息即可
            for doc in docs:               
                try:
                    # 清理原始图片对象（如果存在）
                    if "orig_image" in doc and doc["orig_image"]:
                        doc["orig_image"].close()
                        del doc["orig_image"]
                    
                    # 清理当前图片对象
                    if "image" in doc and not isinstance(doc["image"], bytes):
                        doc["image"].close()
                    del doc["image"]  # 移除图片引用
                
                except Exception:
                    logging.info(f"清理图片引用时发生异常: 文档ID={self.document.id}, 切片ID={doc.get('id', 'unknown')}")
                    pass

            return docs
                
        except Exception as e:
            logging.exception(f"保存切片图片时发生异常: 文档ID={self.document.id}, 切片ID={doc.get('id', 'unknown')}, 错误={e}")
            raise

    async def _process_auto_keywords(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为文档切片自动生成关键词
        
        Args:
            docs: 文档切片列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的文档切片列表，包含关键词信息
        """          
        try:   
            st = timer()
            logging.info("开始为每个切片生成关键词...")
            
            topn = self.parser_config.get("auto_keywords", 3)
            # 并发处理所有文档切片
            for doc in docs:
                cached = await get_llm_cache(
                    self.chat_model.llm_name, 
                    doc["content_with_weight"], 
                    "keywords", 
                    {"topn": topn}
                )

                # 没有缓存则调用模型生成         
                if not cached:
                    async with CHAT_LIMITER:
                        cached = await keyword_extraction(
                            self.chat_model, 
                            doc["content_with_weight"], 
                            topn
                        )
                    
                    await set_llm_cache(
                        self.chat_model.llm_name, 
                        doc["content_with_weight"], 
                        cached, 
                        "keywords", 
                        {"topn": topn}
                    )

                # 保存信息     
                if cached:
                    doc["important_kwd"] = cached.split(",")
                    doc["important_tks"] = rag_tokenizer.tokenize(" ".join(doc["important_kwd"]))
                                
            logging.info(f"关键词生成完成: {len(docs)} 个切片，耗时 {timer() - st:.2f}秒")
            return docs
            
        except Exception as e:
            logging.error(f"为切片生成关键词失败: {doc.get('id', 'unknown')}, 错误: {e}")
            return docs

    async def _process_auto_questions(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为文档切片自动生成问题
        
        Args:
            docs: 文档切片列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的文档切片列表，包含问题信息
        """
        try:
            st = timer()
            logging.info("开始为每个切片生成问题...")
            
            topn = self.parser_config.get("auto_questions", 3)
            # 并发处理所有文档切片
            for doc in docs:
                cached = await get_llm_cache(
                    self.chat_model.llm_name, 
                    doc["content_with_weight"], 
                    "question", 
                    {"topn": topn}
                )
                    
                # 没有缓存，则直接调用模型生成
                if not cached:
                    async with CHAT_LIMITER:
                        cached = await question_proposal(
                            self.chat_model, 
                            doc["content_with_weight"], 
                            topn
                        )
                    await set_llm_cache(
                        self.chat_model.llm_name, 
                        doc["content_with_weight"], 
                        cached, 
                        "question", 
                        {"topn": topn}
                    )
                
                # 保存信息
                if cached:
                    doc["question_kwd"] = cached.split("\n")
                    doc["question_tks"] = rag_tokenizer.tokenize("\n".join(doc["question_kwd"]))

            logging.info(f"问题生成完成: {len(docs)} 个切片，耗时 {timer() - st:.2f}秒")
            return docs

        except Exception as e:
            logging.error(f"为切片生成问题失败: {doc.get('id', 'unknown')}, 错误: {e}")
            return docs

    async def _process_auto_tags(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为文档切片自动生成标签
        """
        try:
            st = timer()
            logging.info("开始为每个切片生成标签...")

            tag_kb_ids = self.parser_config.get("tag_kb_ids", [])
            topn_tags = self.parser_config.get("topn_tags", 3)

            all_tags = get_tags_from_cache(tag_kb_ids)
            if not all_tags:
                all_tags = await RETRIEVALER.all_tags_in_portion(self.kb.tenant_id, tag_kb_ids)
                set_tags_to_cache(tag_kb_ids, all_tags)
            else:
                all_tags = json.loads(all_tags)

            examples = []
            docs_to_tag = []
            S = 1000
            for doc in docs:
                if await RETRIEVALER.tag_content(
                    self.kb.tenant_id, 
                    tag_kb_ids, 
                    doc, 
                    all_tags, 
                    topn_tags=topn_tags, 
                    S=S
                ) and len(doc[TAG_FLD]) > 0:
                    examples.append({"content": doc["content_with_weight"], TAG_FLD: doc[TAG_FLD]})
                else:       
                    docs_to_tag.append(doc)

            for doc in docs_to_tag:              
                cached = await get_llm_cache(
                    self.chat_model.llm_name, 
                    doc["content_with_weight"], 
                    all_tags, 
                    {"topn": topn_tags}
                )

                # 没有缓存，则直接调用模型生成
                if not cached:
                    picked_examples = random.choices(examples, k=2) if len(examples)>2 else examples
                    if not picked_examples:
                        picked_examples.append({"content": "This is an example", TAG_FLD: {'example': 1}})
                    async with CHAT_LIMITER:
                        cached = await content_tagging(
                            self.chat_model, 
                            doc["content_with_weight"], 
                            all_tags, 
                            picked_examples, 
                            topn=topn_tags
                        )
                    
                    if cached:
                        cached = json.dumps(cached)
                
                # 保存信息
                if cached:
                    await set_llm_cache(self.chat_model.llm_name, doc["content_with_weight"], cached, all_tags, {"topn": topn_tags})
                    doc[TAG_FLD] = json.loads(cached)

            logging.info(f"标签生成完成: {len(docs)} 个切片，耗时 {timer() - st:.2f}秒")
            return docs
        except Exception as e:
            logging.error(f"为切片生成标签失败: {doc.get('id', 'unknown')}, 错误: {e}")
            return docs

    async def _process_concept_extraction(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从每个切片中提取名词概念及描述，同名概念合并描述并追加 doc_id。"""
        from app.domains.services.concept_service import ConceptService
        try:
            st = timer()
            logging.info("开始为每个切片提取概念...")
            kb_id = str(self.document.kb_id)

            for doc in docs:
                async with CHAT_LIMITER:
                    concepts = await ConceptService.extract_concepts(self.db_session, kb_id, doc["content_with_weight"])
                if concepts:
                    await ConceptService.create_or_merge(
                        db_session=self.db_session,
                        kb_id=str(self.document.kb_id),
                        doc_id=self.document.id,
                        concepts=concepts,
                    )
            logging.info(f"概念提取完成: {len(docs)} 个切片，耗时 {timer() - st:.2f}秒")
        except Exception as e:
            logging.error(f"为切片提取概念失败: {e}")
        return docs


    async def _embedding_chunks(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        对文档切片进行向量化处理
        
        Args:
            chunks: 文档切片列表
            task: 任务配置
            
        Returns:
            tuple: (token_count, vector_size)
        """
        try:
            # 准备标题和内容文本
            title_texts = []
            content_texts = []
            
            for chunk in chunks:
                # 获取文档标题，默认为"Title"
                title = chunk.get("docnm_kwd", "Title")
                title_texts.append(title)
                
                # 获取内容文本，优先使用问题关键词，否则使用内容权重文本
                question_keywords = chunk.get("question_kwd", [])
                if question_keywords:
                    content = "\n".join(question_keywords)
                else:
                    content = chunk["content_with_weight"]
                
                # 清理HTML标签
                content = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", content)
                if not content:
                    content = "None"
                content_texts.append(content)

            total_token_count = 0
            
            if len(title_texts) == len(content_texts):
                # 只对第一个标题进行向量化
                title_vectors, token_count = await self.embedding_model.encode(title_texts[0: 1])
                # 将第一个标题的向量复制到所有文档
                title_texts = np.concatenate([title_vectors for _ in range(len(title_texts))], axis=0)
                total_token_count += token_count

            # 批量处理内容向量化
            content_vectors = np.array([])
            max_tokens = self.embedding_model.mdl.configs.get("max_tokens", 8192)
            for batch_start in range(0, len(content_texts), EMBEDDING_BATCH_SIZE):
                batch_end = batch_start + EMBEDDING_BATCH_SIZE
                batch_content_texts = content_texts[batch_start:batch_end]
                
                # 截断文本到模型最大长度
                truncated_texts = [
                    truncate(text, max_tokens - 10) 
                    for text in batch_content_texts
                ]
                
                batch_vectors, token_count = await self.embedding_model.encode(truncated_texts)
                
                if len(content_vectors) == 0:
                    content_vectors = batch_vectors
                else:
                    content_vectors = np.concatenate((content_vectors, batch_vectors), axis=0)
                total_token_count += token_count

            content_texts = content_vectors
            
            # 计算最终向量
            filename_embedding_weight = self.parser_config.get("filename_embd_weight", 0.1)
            if not filename_embedding_weight:
                filename_embedding_weight = 0.1
            title_weight = float(filename_embedding_weight)
            
            # 检查title_texts是否已经向量化
            if len(title_texts) == len(content_texts):
                final_vectors = title_weight * title_texts + (1 - title_weight) * content_texts
            else:
                final_vectors = content_texts

            assert len(final_vectors) == len(chunks)

            # 将向量添加到文档切片内容中
            vector_size = 0
            for index, chunk in enumerate(chunks):
                vector = final_vectors[index].tolist()
                vector_size = len(vector)
                # 使用向量长度作为键名
                chunk[f"q_{vector_size}_vec"] = vector
            
            return total_token_count, vector_size
            
        except Exception as e:
            logging.error(f"向量化处理失败: {e}")
            raise

    async def _store_chunks_vector(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        存储向量结果
        
        Args:
            chunks: 切片列表
            
        Returns:
            bool: 存储是否成功
        """

        try: 
            doc_store_result = ""
            for b in range(0, len(chunks), DOC_BULK_SIZE):
                doc_store_result = await self.vector_store.insert(
                    chunks[b:b + DOC_BULK_SIZE], self.kb.tenant_id, self.kb.id
                ) 

                # 如果存储返回错误结果，记录错误信息并抛出异常（doc_store_result为空表示成功，非空表示有错误）
                if doc_store_result:
                    error_message = f"Insert chunk error: {doc_store_result}, please check log file and Elasticsearch/Infinity status!"
                    logging.error(error_message)
                    raise Exception(error_message)
            
            return True
            
        except Exception as e:
            logging.error(f"存储向量结果失败: {e}")
            raise

    async def _execute_raptor_task(self) -> tuple:
        """
        执行RAPTOR任务
            
        Returns:
            tuple: (chunks, token_count)
        """
        try: 
            # 运行RAPTOR
            async with KG_LIMITER:        
                # 检查模型强度
                chat_strong = await self.chat_model.is_strong_enough()
                if not chat_strong:
                    raise Exception("聊天模型强度测试失败，无法执行RAPTOR任务")
                
                embedding_strong = await self.embedding_model.is_strong_enough()
                if not embedding_strong:
                    raise Exception("嵌入模型强度测试失败，无法执行RAPTOR任务")

                vector_size = await self.embedding_model.get_embedding_vector_size()
                vector_name = f"q_{vector_size}_vec"

                chunks = []
                for doc_chunk in await RETRIEVALER.chunk_list(
                    self.document.id, 
                    self.kb.tenant_id,
                    [str(self.document.kb_id)],
                    fields=["content_with_weight", vector_name]
                ):
                    chunks.append((doc_chunk["content_with_weight"], np.array(doc_chunk[vector_name])))

                # 创建RAPTOR实例
                raptor = Raptor(
                    self.parser_config["raptor"].get("max_cluster", 64),
                    self.chat_model,
                    self.embedding_model,
                    self.parser_config["raptor"]["prompt"],
                    self.parser_config["raptor"]["max_token"],
                    self.parser_config["raptor"]["threshold"]
                )
                
                original_length = len(chunks)                
                # 运行RAPTOR
                chunks = await raptor(
                    chunks, 
                    self.parser_config["raptor"]["random_seed"], 
                    self.callback.progress_callback
                )
                
                # 创建文档信息对象
                doc = {
                    "doc_id": self.document.id,
                    "kb_id": [str(self.document.kb_id)],
                    "docnm_kwd": self.document.name,
                    "title_tks": rag_tokenizer.tokenize(self.document.name)
                }
                
                if self.kb.page_rank > 0:
                    doc[PAGERANK_FLD] = int(self.kb.page_rank)
                
                result_chunks = []
                token_count = 0                
                # 处理RAPTOR生成的新chunks
                for content, vector in chunks[original_length:]:
                    chunk_doc = copy.deepcopy(doc)
                    chunk_doc["id"] = xxhash.xxh64((content + str(chunk_doc["doc_id"])).encode("utf-8")).hexdigest()
                    chunk_doc["create_time"] = str(datetime.now()).replace("T", " ")[:19]
                    chunk_doc["create_timestamp_flt"] = datetime.now().timestamp()
                    chunk_doc[vector_name] = vector.tolist()
                    chunk_doc["content_with_weight"] = content
                    chunk_doc["content_ltks"] = rag_tokenizer.tokenize(content)
                    chunk_doc["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(chunk_doc["content_ltks"])
                    result_chunks.append(chunk_doc)
                    token_count += num_tokens_from_string(content)

                # 保存chunks
                await self._store_chunks_vector(result_chunks)
                
                return result_chunks, token_count
            
        except Exception as e:
            logging.error(f"RAPTOR任务执行失败: {e}")
            raise

    async def _execute_graphrag_task(self) -> bool:
        """
        执行GraphRAG任务

        Returns:
            bool: 是否成功
        """
        try:
            # 检查LLM模型是否足够强大
            chat_strong = await self.chat_model.is_strong_enough()
            if not chat_strong:
                raise Exception("聊天模型强度测试失败，无法执行GraphRAG任务")

            with_resolution = self.parser_config.get("graphrag", {}).get("resolution", False)
            with_community = self.parser_config.get("graphrag", {}).get("community", False)

            row = {
                "tenant_id": self.kb.tenant_id,
                "kb_id": self.document.kb_id,
                "doc_id": self.document.id,
                "kb_parser_config": self.parser_config,
            }

            # 运行GraphRAG
            async with KG_LIMITER:          
                 await run_graphrag(
                    row,
                    self.kb.language,
                    with_resolution,
                    with_community,
                    self.chat_model,
                    self.embedding_model,
                    self.callback.progress_callback,    
                )

            return True

        except Exception as e:
            logging.error(f"GraphRAG任务执行失败: {e}")
            return False

