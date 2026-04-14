from typing import Any, Optional
import asyncio
import logging
from openai.lib.azure import AsyncAzureOpenAI
from .base import BaseSTT, MAX_RETRY_ATTEMPTS, CONNECTION_TIMEOUT



class AzureSTT(BaseSTT):
    """Azure OpenAI的语音转文本模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str, **kwargs):
        """
        初始化Azure OpenAI语音转文本模型
        
        Args:
            api_key (str): Azure API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): Azure端点URL
            lang (str): 语言，默认为Chinese
            **kwargs: 其他参数
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        self.client = AsyncAzureOpenAI(
            api_key=api_key, 
            azure_endpoint=base_url, 
            api_version="2024-02-01",
            timeout=CONNECTION_TIMEOUT
        )

    async def stt(self, audio: Any, **kwargs) -> tuple[str, int]:
        """
        将音频转录为文本
        
        Args:
            audio: 音频文件对象
            **kwargs: 其他参数
            
        Returns:
            tuple: (转录文本, 输入音频的token等价量)
            
        Raises:
            Exception: 当API请求失败时
            
        语言支持方式:
            - 通过初始化时的lang参数设置语言
            - 默认设置为"Chinese"
            - 支持多种语言配置
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                transcription = await self.client.audio.transcriptions.create(
                    model=self.model_name, 
                    file=audio, 
                    response_format="text"
                )
                text = transcription.text.strip()
                
                return text, self._total_token_count(transcription, audio)
            
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Azure STT失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Azure STT最终失败: {e}")
                    raise Exception(f"**ERROR**: {e}")