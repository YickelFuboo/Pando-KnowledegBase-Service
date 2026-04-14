import base64
import io
import random
import logging
from abc import ABC
from typing import Tuple, Any, Optional

# 重试配置常量
MAX_RETRY_ATTEMPTS = 3  # 最大尝试次数
RETRY_DELAY = 2  # 重试间隔（秒）
CONNECTION_TIMEOUT = 30  # 连接超时（秒）


class BaseSTT(ABC):
    """语音转文本模型基类，提供语音转文本功能"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化语音转文本模型基类
        
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


    async def stt(self, audio: Any, **kwargs) -> Tuple[str, int]:
        """
        将音频转录为文本
        
        Args:
            audio: 音频文件对象
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, int]: (转录文本, token数量)
        """
        pass


    def _total_token_count(self, response: Any, audio: Any) -> int:
        """
        从响应中提取token总数
        
        Args:
            response: API响应对象
            audio: 音频输入
            
        Returns:
            int: token总数
        """
        try:
            return response.usage.total_tokens
        except Exception:
            pass
        
        try:
            return response["usage"]["total_tokens"]
        except Exception:
            pass

        try:
            return self._calculate_input_tokens(audio)
        except Exception:
            pass

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

    def _calculate_input_tokens(self, audio: Any) -> int:
        """
        计算STT输入音频的token等价量
        业界标准：STT按音频时长计费，需要将时长转换为token等价量
        
        Args:
            audio: 音频文件对象
            
        Returns:
            int: 输入音频的token等价量
        """
        # 基于音频时长计算token等价量
        # 参考OpenAI Whisper: $0.006 per minute = $0.0001 per second
        # 假设1秒音频等价于10个token（可根据实际API调整）
        try:
            # 获取音频大小
            if hasattr(audio, 'read'):
                # 如果是文件对象
                audio.seek(0)
                audio_size = len(audio.read())
                audio.seek(0)
            elif isinstance(audio, bytes):
                # 如果是字节数据
                audio_size = len(audio)
            else:
                return 0
            
            # 基于音频大小估算时长
            # 假设标准音频格式：16kHz采样率，16bit深度，单声道
            # 音频时长(秒) = 音频大小(字节) / (采样率 * 位深度 / 8)
            estimated_duration = audio_size / (16000 * 2)  # 16kHz, 16bit
            
            # 根据时长计算token等价量
            # 业界标准：每秒音频约等于10个token（参考OpenAI定价）
            # 这个比例可以根据具体API提供商调整
            estimated_tokens = max(1, int(estimated_duration * 10))
            
            return estimated_tokens
            
        except Exception:
            # 如果计算失败，使用简单的基于大小的估算
            try:
                if hasattr(audio, 'read'):
                    audio.seek(0)
                    audio_size = len(audio.read())
                    audio.seek(0)
                elif isinstance(audio, bytes):
                    audio_size = len(audio)
                else:
                    return 0
                
                # 备用方案：每3KB音频约等于1个token
                # 这个比例基于音频时长估算的简化版本
                return max(1, audio_size // 3072)
            except Exception:
                return 0


    def audio2base64(self, audio: Any) -> str:
        """
        将音频文件转换为base64编码字符串
        
        Args:
            audio: 音频文件，可以是bytes或BytesIO对象
            
        Returns:
            str: base64编码的音频字符串
            
        Raises:
            TypeError: 当输入音频格式不正确时
        """
        if isinstance(audio, bytes):
            return base64.b64encode(audio).decode("utf-8")
        if isinstance(audio, io.BytesIO):
            return base64.b64encode(audio.getvalue()).decode("utf-8")
        
        raise TypeError("The input audio file should be in binary format.")


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