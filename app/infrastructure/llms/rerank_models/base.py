import os
import re
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
import numpy as np
from ..utils import num_tokens_from_string
from app.config.settings import Settings

# 重试配置常量
MAX_RETRY_ATTEMPTS = 3  # 最大尝试次数
RETRY_DELAY = 2  # 重试间隔（秒）
CONNECTION_TIMEOUT = 30  # 连接超时（秒）


class BaseRank(ABC):
    """重排序模型基类，用于对检索结果进行重新排序"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化重排序模型基类
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (Optional[str]): API基础URL
            kwargs (dict): 其他参数
        """
        self.api_key = api_key
        self.model_provider = model_provider
        self.model_name = model_name
        self.base_url = base_url
        self.configs = kwargs

    async def similarity(self, query: str, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        计算查询与文本列表的相似度分数
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            
        Returns:
            Tuple[np.ndarray, int]: (相似度分数数组, token数量)
        """
        pass

    def _total_token_count(self, respone, texts):
        """
        从响应中提取token总数
        
        Args:
            respone: API响应对象
            texts: 文本列表
            
        Returns:
            int: token总数
        """
        if respone:
            try:
                return respone.usage.total_tokens
            except Exception:
                pass
            
            try:
                return respone["usage"]["total_tokens"]
            except Exception:
                pass

        if texts:
            return sum([num_tokens_from_string(t) for t in texts])
        
        return 0

    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in [
            'connection', 'timeout', 'network', 'temporary', 'busy', 'rate limit', 'overload', '429', '503', '502', '504', '500'
        ])
    
    def _get_delay(self, attempt: int = 0) -> float:
        """获取重试延迟时间（指数退避 + 随机抖动）"""
        # 指数退避：2^attempt * 基础延迟
        base_delay = 1.0
        exponential_delay = base_delay * (2 ** attempt)
        
        # 添加随机抖动，避免雷群效应
        jitter = random.uniform(0.5, 1.5)
        
        # 限制最大延迟为30秒
        max_delay = 30.0
        delay = min(exponential_delay * jitter, max_delay)
        
        return delay

    def _get_model_cache_path(self, model_name: str) -> str:
        """
        获取模型缓存路径
        
        Args:
            model_name (str): 模型名称
            
        Returns:
            str: 模型缓存路径
        """
        settings = Settings()
        
        # 通过查找包含main.py的目录来确定项目根目录
        current_dir = os.path.dirname(__file__)
        project_root = current_dir
        while project_root != os.path.dirname(project_root):  # 直到到达文件系统根目录
            if os.path.exists(os.path.join(project_root, "main.py")):
                break
            project_root = os.path.dirname(project_root)
        
        # 创建缓存目录
        cache_dir = os.path.join(project_root, settings.model_cache_dir, "rerank")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 清理模型名称，移除用户名前缀
        clean_model_name = re.sub(r"^[a-zA-Z0-9]+/", "", model_name)
        model_path = os.path.join(cache_dir, clean_model_name)
        
        return model_path