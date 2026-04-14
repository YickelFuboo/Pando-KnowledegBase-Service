import aiohttp
import asyncio
import logging
from typing import Generator, Optional, Tuple
from .base import BaseTTS, MAX_RETRY_ATTEMPTS


class OpenAITTS(BaseTTS):
    """OpenAI的文本转语音模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "tts-1", base_url: Optional[str] = None, **kwargs):
        """
        初始化OpenAI TTS模型
        
        Args:
            api_key (str): OpenAI API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称，默认为tts-1
            base_url (Optional[str]): API基础URL，默认为OpenAI官方URL
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def tts(self, text: str, **kwargs) -> Tuple[Generator[bytes, None, None], int]:
        """
        将文本转换为语音
        
        Args:
            text (str): 待转换的文本
            voice (str): 声音类型，可选值：alloy, echo, fable, onyx, nova, shimmer
            **kwargs: 其他参数            
                Voice参数说明:
                    - alloy: 中性声音，适合大多数场景
                    - echo: 回声效果，声音较为清晰
                    - fable: 寓言风格，声音较为温和
                    - onyx: 深沉声音，适合正式场合
                    - nova: 年轻声音，适合轻松内容
                    - shimmer: 闪烁效果，声音较为活泼
            
        Returns:
            Tuple[Generator[bytes, None, None], int]: (音频数据生成器, token数量)
            
        Raises:
            Exception: 当API请求失败时

        """
        # 默认声音为alloy
        voice = "alloy"
        if kwargs.get("voice"):
            voice = kwargs.get("voice") 

        text = self.normalize_text(text)
        payload = {
            "model": self.model_name,
            "voice": voice,
            "input": text
        }

        # 计算输入文本的token数量
        input_tokens = self._total_token_count(text)

        async def audio_generator():
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/audio/speech",
                            headers=self.headers,
                            json=payload,
                            stream=True
                        ) as response:
                            if response.status != 200:
                                raise Exception(f"**Error**: {response.status}, {await response.text()}")
                            
                            async for chunk in response.content.iter_chunked(1024):
                                if chunk:
                                    yield chunk
                            return  # 成功，退出重试循环
                            
                except Exception as e:
                    if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                        delay = self._get_delay(attempt)
                        logging.warning(f"OpenAI TTS失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logging.error(f"OpenAI TTS最终失败: {e}")
                        raise Exception(f"**ERROR**: {e}")

        return audio_generator(), input_tokens