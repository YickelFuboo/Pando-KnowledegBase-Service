import aiohttp
import asyncio
import logging
from typing import Generator, Optional, Tuple
from .base import BaseTTS, MAX_RETRY_ATTEMPTS


class SiliconFlowTTS(BaseTTS):
    """SiliconFlow的文本转语音模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "FunAudioLLM/CosyVoice2-0.5B", base_url: Optional[str] = None, **kwargs):
        """
        初始化SiliconFlow TTS模型
        
        Args:
            api_key (str): SiliconFlow API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (Optional[str]): API基础URL，默认为SiliconFlow官方URL
        """
        if not base_url:
            base_url = "https://api.siliconflow.cn/v1"
        
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
            voice (str): 声音类型
            **kwargs: 其他参数        
                Voice参数说明:
                    - anna: 默认女性声音，适合大多数场景
                    - 支持多种声音选择，包括不同语言和风格
                    - 具体可用声音取决于SiliconFlow服务配置
                    - 可通过API文档查看所有可用的声音选项
            
        Returns:
            Tuple[Generator[bytes, None, None], int]: (音频数据生成器, token数量)
            
        Raises:
            Exception: 当API请求失败时
        """
        voice = "anna"
        if kwargs.get("voice"):
            voice = kwargs.get("voice") 

        text = self.normalize_text(text)
        payload = {
            "model": self.model_name,
            "input": text,
            "voice": f"{self.model_name}:{voice}",
            "response_format": "mp3",
            "sample_rate": 123,
            "stream": True,
            "speed": 1,
            "gain": 0,
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
                        logging.warning(f"SiliconFlow TTS失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logging.error(f"SiliconFlow TTS最终失败: {e}")
                        raise Exception(f"**ERROR**: {e}")

        return audio_generator(), input_tokens