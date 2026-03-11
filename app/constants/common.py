from enum import Enum, StrEnum


class KBConstants:
    """知识库相关常量"""
    
    # 名称长度限制
    NAME_MAX_LENGTH = 128
    
    # 描述长度限制
    DESCRIPTION_MAX_LENGTH = 1000
    
    # 默认分页大小
    DEFAULT_PAGE_SIZE = 20
    
    # 最大分页大小
    MAX_PAGE_SIZE = 100
    
    # 支持的文档类型
    SUPPORTED_DOCUMENT_TYPES = [
        "pdf", "doc", "docx", "txt", "md", "html", "htm"
    ]

    # 支持的图像类型
    SUPPORTED_IMAGE_TYPES = [
        "jpg", "jpeg", "png", "gif", "svg"
    ]

    # 图像大小限制 (MB)
    MAX_IMAGE_SIZE = 10
    
    # 默认解析器配置
    DEFAULT_PARSER_CONFIG = {
        "pages": [[1, 1000000]],
        "chunk_token_num": 512,
        "delimiter": "\\n!?。；！？",
        "task_page_size": 12,
        "layout_recognize": "DeepDOC",
        "auto_keywords": 0,
        "auto_concepts": False,
        "auto_questions": 0,
        "html4excel": False,
        "topn_tags": 3,
        "tag_kb_ids": [],
        "filename_embd_weight": 0.1,
        "raptor": {
            "use_raptor": False,
            "prompt": "请总结以下段落。 小心数字，不要编造。 段落如下：\n{cluster_content}\n以上就是你需要总结的内容。",
            "max_token": 256,
            "threshold": 0.1,
            "max_cluster": 64,
            "random_seed": 0
        },
        "graphrag": {
            "use_graphrag": False,
            "entity_types": [
                "organization",
                "person",
                "geo",
                "event",
                "category"
            ],
            "method": "light",
            "community": False,
            "resolution": False
        }
    }
    
    # 解析器配置字段说明
    PARSER_CONFIG_FIELD_DESCRIPTIONS = {
        "pages": {"type": "array", "description": "页面范围", "default": "[[1, 1000000]]"},
        "chunk_token_num": {"type": "integer", "description": "每块的最大token数量", "range": "1-2048", "default": 512},
        "task_page_size": {"type": "integer", "description": "每个任务处理的页面数量", "range": "1-128", "default": 12},
        "layout_recognize": {"type": "string", "description": "布局识别方式", "options": ["DeepDOC", "Plain Text"], "default": "DeepDOC"},
        "auto_keywords": {"type": "integer", "description": "自动生成关键词数量", "range": "0-32", "default": 0},
        "auto_concepts": {"type": "boolean", "description": "是否自动提取概念", "default": False},
        "auto_questions": {"type": "integer", "description": "自动生成问题数量", "range": "0-10", "default": 0},
        "html4excel": {"type": "boolean", "description": "Excel是否转换为HTML", "default": False},
        "topn_tags": {"type": "integer", "description": "标签数量", "range": "1-10", "default": 3},
        "tag_kb_ids": {"type": "array", "description": "标签来源知识库ID列表", "default": "[]", "note": "用于指定从哪些知识库中获取标签"},
        "filename_embd_weight": {"type": "float", "description": "文件名在向量化中的权重", "range": "0.0-1.0", "default": 0.1},
        "raptor": {
            "use_raptor": {"type": "boolean", "description": "是否启用RAPTOR", "default": False},
            "prompt": {"type": "string", "description": "RAPTOR使用的提示词", "default": "请总结以下段落。 小心数字，不要编造。 段落如下：\n{cluster_content}\n以上就是你需要总结的内容。"},
            "max_token": {"type": "integer", "description": "最大token数", "range": "0-2048", "default": 256},
            "threshold": {"type": "float", "description": "阈值", "range": "0.0-1.0", "default": 0.1},
            "max_cluster": {"type": "integer", "description": "最大聚类数", "range": "1-1024", "default": 64},
            "random_seed": {"type": "integer", "description": "随机种子", "range": ">=0", "default": 0}
        },
        "graphrag": {
            "use_graphrag": {"type": "boolean", "description": "是否启用GraphRAG", "default": False},
            "entity_types": {"type": "array", "description": "实体类型列表", "default": ["organization", "person", "geo", "event", "category"]},
            "method": {"type": "string", "description": "方法类型", "options": ["light", "general"], "default": "light"},
            "community": {"type": "boolean", "description": "是否启用社区检测", "default": False},
            "resolution": {"type": "boolean", "description": "是否启用分辨率优化", "default": False}
        }
    }


class DocumentConstants:
    """文件相关常量"""
    
    # 文件名称长度限制
    FILE_NAME_LEN_LIMIT = 255

        
    # 文档大小限制 (MB)
    MAX_DOCUMENT_FILE_SIZE = 128 * 1024 * 1024



class TenantConstants:
    """租户相关常量"""
    
    # 租户名称长度限制
    TENANT_NAME_MAX_LENGTH = 128
    
    # 租户描述长度限制
    TENANT_DESCRIPTION_MAX_LENGTH = 1000


class ContentType(StrEnum):
    """内容类型常量"""
    
    JSON = "application/json"
    FORM_DATA = "multipart/form-data"
    FORM_URLENCODED = "application/x-www-form-urlencoded"
    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    TEXT_XML = "text/xml"
    APPLICATION_XML = "application/xml"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_ZIP = "application/zip"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_GIF = "image/gif"
    IMAGE_SVG = "image/svg+xml"


class FileExtension(StrEnum):
    """文件扩展名常量"""
    
    # 文档类型
    PDF = ".pdf"
    DOC = ".doc"
    DOCX = ".docx"
    TXT = ".txt"
    MD = ".md"
    HTML = ".html"
    HTM = ".htm"
    
    # 图片类型
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    GIF = ".gif"
    SVG = ".svg"
    
    # 压缩类型
    ZIP = ".zip"
    RAR = ".rar"
    TAR = ".tar"
    GZ = ".gz"
    
    # 音频类型
    MP3 = ".mp3"
    WAV = ".wav"
    FLAC = ".flac"
    
    # 视频类型
    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    WMV = ".wmv" 

class FileType(StrEnum):
    PDF = 'pdf'
    DOC = 'doc'
    VISUAL = 'visual'
    AURAL = 'aural'
    VIRTUAL = 'virtual'
    FOLDER = 'folder'
    OTHER = "other"

class FileSource(StrEnum):
    """文件来源枚举"""
    KNOWLEDGEBASE = "knowledgebase"
    UPLOAD = "upload"
    WEB_CRAWL = "web_crawl"

class PDFParser(StrEnum):
    """PDF解析器枚举"""
    DEEPDOC = "DeepDoc"
    NATIVE = "Native"


class KnowledgeGraphMethod(StrEnum):
    """知识图谱方法枚举"""
    GENERAL = "general"
    LIGHT = "light"