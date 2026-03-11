import sys
import io
import re
import threading
import uuid
import asyncio
import logging
import pdfplumber
import aspose.pydrawing as drawing
import aspose.slides as slides
import concurrent.futures
from io import BytesIO
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
from sqlalchemy import select, and_, or_, func, desc, asc
from pathlib import Path
from app.domains.models import Document, KB, ProcessStatus
from app.rag_core.utils import ParserType
from app.constants.common import DocumentConstants, FileType, FileSource
from app.domains.schemes.document import FileUploadResult
from app.domains.services.common.file_service import FileService, FileUsage
from app.domains.services.common.doc_vector_store_service import DOC_STORE_CONN
from app.infrastructure.database import get_db


class DocumentService:
    """文档服务类"""

    @staticmethod
    async def upload_document_to_kb(
        session: AsyncSession,
        kb: KB,
        file: UploadFile,
        created_by: str
    ) -> FileUploadResult:
        """上传单个文档到知识库"""

        try:
            filename = file.filename    
            file_obj = file.file

            # 读取文件内容
            file_content = file_obj.read()
            file_size = len(file_content)
            
            # 验证文件名有效性
            is_valid, error_msg = FileService.validate_filename(filename)
            if not is_valid:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件 '{filename}': {error_msg}"
                )
            
            # 处理重复文件名
            final_filename = await DocumentService._handle_duplicate_filename(session, kb.id, filename)

            # 判断文件大小
            if file_size > DocumentConstants.MAX_DOCUMENT_FILE_SIZE:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件大小 {file_size} 超过最大限制 {DocumentConstants.MAX_DOCUMENT_FILE_SIZE} M"
                )
            
            # 识别文件类型
            file_type = DocumentService._get_file_type_by_suffix(final_filename)
            if file_type == FileType.OTHER:
                return FileUploadResult(
                    filename=filename,
                    success=False,
                    error=f"文件 '{filename}': 不支持的文件类型"
                )
            
            # 特殊处理PDF文件
            if file_type == FileType.PDF:
                file_content = FileService.fix_broken_pdf(file_content)
            
            # 存储文件到对象存储
            file_id = await FileService.upload_file(
                file_data=io.BytesIO(file_content),
                file_name=final_filename,
                file_usage=FileUsage.DOCUMENT
            )
            
            # 生成缩略图
            thumbnail_id = await DocumentService._generate_document_thumbnail(
                final_filename, file_content
            )
            
            # 确定解析器类型
            parser_type = DocumentService._get_parser_type(file_type, final_filename, kb.parser_id)
            
            # 创建文档记录
            document = Document(
                id=str(uuid.uuid4()),
                kb_id=kb.id,
                name=final_filename,
                type=file_type.value,
                suffix=Path(final_filename).suffix.lstrip("."), # 文件扩展名
                file_id=file_id,
                size=file_size,
                parser_id=parser_type.value,
                parser_config=kb.parser_config,
                source_type=FileSource.UPLOAD,
                created_by=created_by,
                thumbnail_id=thumbnail_id,
                process_status=ProcessStatus.INIT
            )
            
            session.add(document)
            await session.commit()
            await session.refresh(document)            
            await DocumentService._update_kb_doc_count(session, kb.id)
            
            logging.info(f"文档上传成功: {final_filename}")
            
            return FileUploadResult(
                filename=filename,
                success=True,
                document_id=document.id,
            )
            
        except Exception as e:
            error_msg = f"处理文件 '{filename}' 失败: {str(e)}"
            logging.error(error_msg)
            return FileUploadResult(
                filename=filename,
                success=False,
                error=error_msg
            )
    
    @staticmethod
    async def _handle_duplicate_filename(
        session: AsyncSession, 
        kb_id: str, 
        filename: str
    ) -> str:
        """处理重复文件名"""
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        counter = 1
        final_filename = filename
        
        while await DocumentService.get_document_by_name(session, kb_id, final_filename):
            final_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        
        return final_filename
    
    @staticmethod
    def _get_file_type_by_suffix(filename: str) -> FileType:
        """根据文件名获取文件类型"""
        filename = filename.lower()
        if re.match(r".*\.pdf$", filename):
            return FileType.PDF

        if re.match(r".*\.(eml|doc|docx|ppt|pptx|yml|xml|htm|json|csv|txt|ini|xls|xlsx|wps|rtf|hlp|pages|numbers|key|md|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|html|sql)$", filename):
            return FileType.DOC

        if re.match(r".*\.(wav|flac|ape|alac|wavpack|wv|mp3|aac|ogg|vorbis|opus)$", filename):
            return FileType.AURAL

        if re.match(r".*\.(jpg|jpeg|png|tif|gif|pcx|tga|exif|fpx|svg|psd|cdr|pcd|dxf|ufo|eps|ai|raw|WMF|webp|avif|apng|icon|ico|mpg|mpeg|avi|rm|rmvb|mov|wmv|asf|dat|asx|wvx|mpe|mpa|mp4)$", filename):
            return FileType.VISUAL

        return FileType.OTHER
    
    @staticmethod
    def _get_parser_type(
        file_type: FileType, 
        filename: str, 
        default_parser: str 
    ) -> ParserType:
        """根据文件类型确定解析器类型"""
        if file_type == FileType.VISUAL:
            return ParserType.PICTURE
        if file_type == FileType.AURAL:
            return ParserType.AUDIO
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION
        if re.search(r"\.(eml)$", filename):
            return ParserType.EMAIL
        return ParserType(default_parser)
    
    @staticmethod
    async def _generate_document_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成文档缩略图"""
        try:
            if re.match(r".*\.pdf$", filename):
                return await DocumentService._generate_pdf_thumbnail(filename, content)
            elif re.match(r".*\.(jpg|jpeg|png|tif|gif|icon|ico|webp)$", filename):
                return await DocumentService._generate_image_thumbnail(filename, content)
            elif re.match(r".*\.(ppt|pptx)$", filename):
                return await DocumentService._generate_ppt_thumbnail(filename, content)
            else:
                return None
                
        except Exception as e:
            logging.error(f"生成文档缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_pdf_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成PDF缩略图"""
        try:
            # 导入必要的库
            
            # 创建全局锁来避免pdfplumber的并发问题
            LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
            if LOCK_KEY_pdfplumber not in sys.modules:
                sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()
            
            with sys.modules[LOCK_KEY_pdfplumber]:
                # 打开PDF文件
                pdf = pdfplumber.open(BytesIO(content))
                
                if not pdf.pages:
                    pdf.close()
                    logging.warning(f"PDF文件 {filename} 没有页面")
                    return None
                
                # 生成缩略图
                buffered = BytesIO()
                resolution = 32
                img = None
                
                for _ in range(10):
                    try:
                        # 使用第一页生成缩略图
                        page = pdf.pages[0]
                        page_image = page.to_image(resolution=resolution)
                        page_image.save(buffered, format="PNG")
                        img = buffered.getvalue()
                        
                        # 如果图片大小合适，则使用当前分辨率
                        if len(img) >= 64000 and resolution >= 2:
                            resolution = resolution / 2
                            buffered = BytesIO()
                        else:
                            break
                    except Exception as e:
                        logging.warning(f"生成PDF缩略图时分辨率 {resolution} 失败: {e}")
                        resolution = resolution / 2
                        buffered = BytesIO()
                        if resolution < 1:
                            break
                
                pdf.close()
                
                if img:
                    # 将缩略图保存到文件存储
                    thumbnail_filename = f"{Path(filename).stem}_thumb.png"
                    
                    # 上传缩略图到对象存储，返回缩略图ID
                    thumbnail_id = await FileService.upload_file(
                        file_data=io.BytesIO(img),
                        file_name=thumbnail_filename,
                        file_usage=FileUsage.DOCUMENT_THUMBNAIL
                    )
                    
                    logging.info(f"PDF缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
                    return thumbnail_id
                else:
                    logging.warning(f"PDF缩略图生成失败: {filename}")
                    return None
                    
        except Exception as e:
            logging.error(f"生成PDF缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_image_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成图片缩略图"""
        try:
            # 导入必要的库
            
            # 打开图片
            image = Image.open(BytesIO(content))
            
            # 转换为RGB模式（如果是RGBA等）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 生成缩略图，保持宽高比
            image.thumbnail((300, 300))
            
            # 保存缩略图到内存
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_content = buffered.getvalue()
            
            # 上传缩略图到对象存储，返回缩略图ID
            thumbnail_filename = f"{Path(filename).stem}_thumb.png"
            
            thumbnail_id = await FileService.upload_file(
                file_data=io.BytesIO(img_content),
                file_name=thumbnail_filename,
                file_usage=FileUsage.DOCUMENT_THUMBNAIL
            )
            
            logging.info(f"图片缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
            return thumbnail_id
            
        except Exception as e:
            logging.error(f"生成图片缩略图失败: {filename}, 错误: {e}")
            return None
    
    @staticmethod
    async def _generate_ppt_thumbnail(
        filename: str,
        content: bytes
    ) -> Optional[str]:
        """生成PPT缩略图"""
        try:
            # 导入必要的库
            
            try:
                with slides.Presentation(BytesIO(content)) as presentation:
                    if not presentation.slides:
                        logging.warning(f"PPT文件 {filename} 没有幻灯片")
                        return None
                    
                    buffered = BytesIO()
                    scale = 0.03
                    img = None
                    
                    for _ in range(10):
                        # https://reference.aspose.com/slides/python-net/aspose.slides/slide/get_thumbnail/#float-float
                        presentation.slides[0].get_thumbnail(scale, scale).save(buffered, drawing.imaging.ImageFormat.png)
                        img = buffered.getvalue()
                        if len(img) >= 64000:
                            scale = scale / 2.0
                            buffered = BytesIO()
                        else:
                            break
                    
                    if img:
                        # 上传缩略图到对象存储，返回缩略图ID
                        thumbnail_filename = f"{Path(filename).stem}_thumb.png"
                        
                        thumbnail_id = await FileService.upload_file(
                            file_data=io.BytesIO(img),
                            file_name=thumbnail_filename,
                            file_usage=FileUsage.DOCUMENT_THUMBNAIL
                        )
                        
                        logging.info(f"PPT缩略图生成成功: {filename} -> 缩略图ID: {thumbnail_id}")
                        return thumbnail_id
                    else:
                        logging.warning(f"PPT缩略图生成失败: {filename}")
                        return None
                        
            except Exception as e:
                logging.warning(f"使用Aspose.Slides生成PPT缩略图失败: {e}")
                return None

        except Exception as e:
            logging.error(f"生成PPT缩略图失败: {filename}, 错误: {e}")
            return None

    @staticmethod
    async def get_documents_by_kb_id(
        session: AsyncSession,
        kb_id: str,
        page: int = 1,
        page_size: int = 20,
        keywords: str = "",
        order_by: str = "created_at",
        desc_order: bool = True
    ) -> Tuple[List[Document], int]:
        """根据知识库ID获取文档列表"""
        try:
            query = select(Document).where(Document.kb_id == kb_id)
            
            if keywords:
                query = query.where(
                    or_(
                        Document.name.contains(keywords),
                        Document.description.contains(keywords)
                    )
                )
            
            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total = count_result.scalar()
            
            if hasattr(Document, order_by):
                order_column = getattr(Document, order_by)
                if desc_order:
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
            else:
                if desc_order:
                    query = query.order_by(desc(Document.created_at))
                else:
                    query = query.order_by(asc(Document.created_at))
            
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            result = await session.execute(query)
            documents = result.scalars().all()
            
            return list(documents), total
            
        except Exception as e:
            logging.error(f"获取文档列表失败: {e}")
            raise
    
    @staticmethod
    async def update_document(
        session: AsyncSession,
        doc_id: str,
        description: Optional[str] = None
    ) -> Document:
        """更新文档信息"""
        try:
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                raise ValueError("文档不存在")

            if description is not None:
                document.description = description
            
            await session.commit()
            await session.refresh(document)
            
            logging.info(f"更新文档成功: {doc_id}")
            return document
            
        except Exception as e:
            await session.rollback()
            logging.error(f"更新文档失败: {e}")
            raise

    @staticmethod
    async def update_document_file(
        session: AsyncSession,
        doc_id: str,
        file: UploadFile,
        created_by: str = None
    ) -> Document:
        """通过替换文件来更新文档"""
        try:
            # 1. 获取原文档信息
            old_document = await DocumentService.get_document_by_id(session, doc_id)
            if not old_document:
                raise ValueError("文档不存在")
            
            # 2. 获取知识库信息
            from app.domains.services.kb_service import KBService
            kb = await KBService.get_kb_by_id(session, old_document.kb_id)
            if not kb:
                raise ValueError("知识库不存在")
            
            # 3. 先删除原文档及相关数据
            await DocumentService.delete_document_by_id(session, doc_id)
            
            # 4. 创建新文档
            result = await DocumentService.upload_document_to_kb(
                session=session,
                kb=kb,
                file=file,
                created_by=created_by
            )
            if not result.success or not result.document_id:
                raise ValueError(result.error)

            # 5. 获取新文档并更新描述
            new_document = await DocumentService.get_document_by_id(session, result.document_id)
            if not new_document:
                raise ValueError("新文档创建失败")

            # 6. 更新新文档的描述与原文档一致
            new_document.description = old_document.description
            await session.commit()
            await session.refresh(new_document)
            
            logging.info(f"更新文档文件成功: {doc_id}")
            return new_document
            
        except Exception as e:
            await session.rollback()
            logging.error(f"更新文档文件失败: {e}")
            raise
    
    @staticmethod
    async def delete_document_by_id(
        session: AsyncSession,
        doc_id: str
    ):
        """删除文档及其相关数据"""
        try:
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                raise ValueError("文档不存在")
            
            # 1. 删除向量存储中的chunks
            await DocumentService.delete_document_chunks(session, doc_id)
            
            # 2. 删除文件存储中的文件
            if document.file_id:
                try:
                    await FileService.delete_file(document.file_id, FileUsage.DOCUMENT)
                    logging.info(f"删除文档文件成功: {document.file_id}")
                except Exception as e:
                    logging.error(f"删除文档文件失败: {e}")
            
            # 3. 删除缩略图
            if document.thumbnail_id:
                try:
                    await FileService.delete_file(document.thumbnail_id, FileUsage.DOCUMENT_THUMBNAIL)
                    logging.info(f"删除缩略图成功: {document.thumbnail_id}")
                except Exception as e:
                    logging.error(f"删除缩略图失败: {e}")
            
            # 4. 删除文档记录
            await session.delete(document)
            await session.commit()
            
            # 5. 更新知识库文档数量
            await DocumentService._update_kb_doc_count(session, document.kb_id)
            
            logging.info(f"删除文档及相关数据成功: {doc_id}")
            
        except Exception as e:
            await session.rollback()
            logging.error(f"删除文档及相关数据失败: {e}")
            raise
    
    @staticmethod
    async def get_document_by_id(
        session: AsyncSession, 
        doc_id: str
    ) -> Optional[Document]:
        """根据ID获取文档"""
        try:
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取文档失败: {e}")
            return None
    
    @staticmethod
    async def get_documents_by_ids(
        session: AsyncSession,
        doc_ids: List[str]
    ) -> List[Document]:
        """根据ID列表获取文档"""
        try:
            result = await session.execute(
                select(Document).where(Document.id.in_(doc_ids))
            )
            return result.scalars().all()
        except Exception as e:
            logging.error(f"批量获取文档失败: {e}")
            return []
    
    @staticmethod
    async def get_document_by_name(
        session: AsyncSession, 
        kb_id: str, 
        name: str
    ) -> Optional[Document]:
        """根据知识库ID和文档名称获取文档"""
        try:
            result = await session.execute(
                select(Document).where(
                    and_(Document.kb_id == kb_id, Document.name == name)
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logging.error(f"获取文档失败: {e}")
            return None

    @staticmethod
    async def _update_kb_doc_count(session: AsyncSession, kb_id: str):
        """更新知识库文档数量"""
        try:
            count_result = await session.execute(
                select(func.count(Document.id)).where(Document.kb_id == kb_id)
            )
            doc_count = count_result.scalar()
            
            kb_result = await session.execute(
                select(KB).where(KB.id == kb_id)
            )
            kb = kb_result.scalar_one_or_none()
            
            if kb:
                kb.doc_num = doc_count
                await session.commit()
                
        except Exception as e:
            logging.error(f"更新知识库文档数量失败: {e}")
            await session.rollback() 

    @staticmethod
    async def get_document_chunk_count(
        session: AsyncSession, 
        doc_id: str
    ) -> int:
        """获取文档的切片数量"""
        try:
            # 获取文档信息以获取kb_id
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                return 0
            
            # 获取知识库信息以获取tenant_id
            from app.domains.services.kb_service import KBService
            kb = await KBService.get_kb_by_id(session, document.kb_id)
            if not kb:
                return 0
            
            tenant_id = kb.tenant_id
            
            # 构建查询条件
            condition = {"doc_id": doc_id}
            
            # 使用向量存储服务进行聚合查询
            result = await DOC_STORE_CONN.search(
                selectFields=["doc_id"],
                highlightFields=[],
                condition=condition,
                matchExprs=[],
                orderBy=None,
                offset=0,
                limit=0,  # 只获取总数，不获取具体数据
                tenant_ids=tenant_id,
                kb_ids=[document.kb_id],
                aggFields=["doc_id"]  # 聚合doc_id字段来获取数量
            )
            
            # 从聚合结果中获取总数
            total = DOC_STORE_CONN.getTotal(result)
            return total
            
        except Exception as e:
            logging.warning(f"获取文档切片数量失败: {e}, doc_id: {doc_id}")
            return 0

    @staticmethod
    async def get_document_chunks(
        session: AsyncSession,
        doc_id: str,
        with_vector: bool = False,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict], int]:
        """获取文档的切片列表"""
        try:
            # 获取文档信息以获取kb_id
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                return [], 0
            
            # 获取知识库信息以获取tenant_id
            from app.domains.services.kb_service import KBService
            kb = await KBService.get_kb_by_id(session, document.kb_id)
            if not kb:
                return [], 0
            
            tenant_id = kb.tenant_id
            
            # 构建查询条件
            condition = {"doc_id": doc_id}
            
            # 计算分页参数
            offset = (page - 1) * page_size
            
            # 使用向量存储服务进行查询
            result = await DOC_STORE_CONN.search(
                selectFields=[],  # 空列表表示返回所有字段
                highlightFields=[],
                condition=condition,
                matchExprs=[],
                orderBy=None,  # 可以添加排序逻辑
                offset=offset,
                limit=page_size,
                tenant_ids=tenant_id,
                kb_ids=[document.kb_id],
                aggFields=[]
            )
            
            # 获取总数
            total = DOC_STORE_CONN.getTotal(result)
            
            # 获取chunk数据
            chunks = DOC_STORE_CONN.getSource(result)

            # 清理向量字段为空
            if not with_vector:
                chunks = DocumentService._clean_chunk_fields(chunks)

            return chunks, total
            
        except Exception as e:
            logging.error(f"获取文档切片列表失败: {e}")
            raise

    @staticmethod
    async def get_documents_chunks(
        session: AsyncSession,
        doc_ids: List[str],
        with_vector: bool = False,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Dict], int]:
        """批量获取多个文档的切片列表"""
        try:
            if not doc_ids:
                return [], 0
            
            # 检查所有文档是否属于同一个知识库
            documents = []
            kb_ids = set()
            
            for doc_id in doc_ids:
                document = await DocumentService.get_document_by_id(session, doc_id)
                if not document:
                    raise ValueError(f"文档 {doc_id} 不存在")
                documents.append(document)
                kb_ids.add(document.kb_id)
            
            # 检查是否所有文档都在同一个知识库中
            if len(kb_ids) > 1:
                raise ValueError(f"文档必须属于同一个知识库，当前文档分布在 {len(kb_ids)} 个不同的知识库中")
            
            # 获取知识库信息以获取tenant_id
            from app.domains.services.kb_service import KBService
            kb = await KBService.get_kb_by_id(session, list(kb_ids)[0])
            if not kb:
                raise ValueError("知识库不存在")
            
            tenant_id = kb.tenant_id
            
            # 构建查询条件 - 使用doc_id的in查询
            condition = {"doc_id": doc_ids}
            
            # 计算分页参数
            offset = (page - 1) * page_size
            
            # 使用向量存储服务进行查询
            result = await DOC_STORE_CONN.search(
                selectFields=[],  # 空列表表示返回所有字段
                highlightFields=[],
                condition=condition,
                matchExprs=[],
                orderBy=None,  # 可以添加排序逻辑
                offset=offset,
                limit=page_size,
                tenant_ids=tenant_id,
                kb_ids=[list(kb_ids)[0]],
                aggFields=[]
            )
            
            # 获取总数
            total = DOC_STORE_CONN.getTotal(result)
            
            # 获取chunk数据
            chunks = DOC_STORE_CONN.getSource(result)
            
            # 清理向量字段为空
            if not with_vector:
                chunks = DocumentService._clean_chunk_fields(chunks)
            
            return chunks, total
            
        except Exception as e:
            logging.error(f"批量获取文档切片列表失败: {e}")
            raise

    @staticmethod
    async def delete_document_chunks(session: AsyncSession, doc_id: str):
        """删除文档的切片数据"""
        try:
            # 获取文档信息以获取kb_id和tenant_id
            document = await DocumentService.get_document_by_id(session, doc_id)
            if not document:
                logging.warning(f"文档 {doc_id} 不存在，跳过删除chunks")
                return
            
            # 获取知识库信息以获取tenant_id
            from app.domains.services.kb_service import KBService
            kb = await KBService.get_kb_by_id(session, document.kb_id)
            if not kb:
                logging.warning(f"知识库 {document.kb_id} 不存在，跳过删除chunks")
                return
            
            tenant_id = kb.tenant_id
            
            # 构建删除条件
            condition = {"doc_id": doc_id}
            
            # 使用向量存储服务删除chunks
            deleted_count = await DOC_STORE_CONN.delete(
                condition=condition,
                tenant_id=tenant_id,
                kb_id=document.kb_id
            )
            logging.info(f"删除文档 {doc_id} 的切片数据，共删除 {deleted_count} 个chunks")

            # 移除Doc相关概念
            from app.domains.services.concept_service import ConceptService
            try:
                await ConceptService.remove_doc(db_session=session, kb_id=document.kb_id, doc_id=doc_id)
            except Exception as ce:
                logging.warning(f"移除文档 {doc_id} 与概念的关联失败: {ce}")
            
        except Exception as e:
            logging.error(f"删除文档切片数据失败: {e}")
            raise  # 删除失败可能是数据不存在

    @staticmethod
    async def update_document_status(session: AsyncSession, doc_id: str, status: ProcessStatus):
        """更新文档状态"""
        try:
            document = await DocumentService.get_document_by_id(session, doc_id)
            if document:
                document.process_status = status
                await session.commit()
                await session.refresh(document)
        except Exception as e:
            logging.error(f"更新文档状态失败: {doc_id}, {status}, {e}")
            await session.rollback()
            return False
        return True

    @staticmethod
    def _clean_chunk_fields(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清理chunk数据中的向量字段，将q_*_vec相关字段置空
        
        Args:
            chunks: 原始chunk数据列表
            
        Returns:
            List[Dict[str, Any]]: 清理后的chunk数据列表
        """
        import re
        
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.copy()  # 复制原始数据
            
            # 将q_*_vec相关字段置空
            for key in cleaned_chunk.keys():
                if re.match(r'q_\d+_vec', key):  # 匹配q_数字_vec格式的字段
                    cleaned_chunk[key] = None
            
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    @staticmethod
    async def update_document_meta_fields(
        session: AsyncSession,
        doc_id: str,
        meta_fields: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """更新文档的元数据字段"""
        try:
            # 获取文档
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            document = result.scalar_one_or_none()
            
            if not document:
                logging.error(f"文档 {doc_id} 不存在，跳过更新meta_fields")
                raise ValueError(f"文档 {doc_id} 不存在")
            
            # 更新meta_fields
            document.meta_fields = meta_fields
            await session.commit()
            await session.refresh(document)
            
            return document
        except Exception as e:
            await session.rollback()
            logging.error(f"更新文档元数据字段失败: {e}")
            raise