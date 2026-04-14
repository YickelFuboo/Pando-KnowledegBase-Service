import re
import random
import logging
from abc import ABC, abstractmethod
from typing import Generator, Any, Optional, Tuple
from ..utils import num_tokens_from_string

# 重试配置常量
MAX_RETRY_ATTEMPTS = 3  # 最大尝试次数
RETRY_DELAY = 2  # 重试间隔（秒）
CONNECTION_TIMEOUT = 30  # 连接超时（秒）


class BaseTTS(ABC):
    """文本转语音模型基类，提供TTS功能"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化文本转语音模型基类
        
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

    @abstractmethod
    async def tts(self, text: str, **kwargs) -> Tuple[Generator[bytes, None, None], int]:
        """
        将文本转换为语音
        
        Args:
            text (str): 待转换的文本
            **kwargs: 其他参数，如voice、format、language等
            
        Returns:
            Tuple[Generator[bytes, None, None], int]: (音频数据生成器, token数量)
        """
        pass

    def _normalize_text(self, text: str) -> str:
        """
        标准化文本，移除特殊标记
        
        Args:
            text (str): 待标准化的文本
            
        Returns:
            str: 标准化后的文本
        """
        # 移除常见的特殊标记
        text = re.sub(r"(\*\*|##\d+\$\$|#)", "", text)
        # 移除多余的空白字符
        text = re.sub(r"\s+", " ", text).strip()
        return text

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
    
    def _total_token_count(self, text: str) -> int:
        """
        从响应中提取token总数
        
        Args:
            text: 文本
            
        Returns:
            int: token总数
        """
        return num_tokens_from_string(text)


    def get_model_info(self) -> dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            dict[str, Any]: 模型信息字典
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url
        }