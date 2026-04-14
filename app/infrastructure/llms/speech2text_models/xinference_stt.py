import os
import aiohttp
import asyncio
import logging
from typing import Any, Optional
from .base import BaseSTT, MAX_RETRY_ATTEMPTS


class XinferenceSTT(BaseSTT):
    """Xinference的语音转文本模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "whisper-small", base_url: Optional[str] = None, **kwargs):
        """
        初始化Xinference语音转文本模型
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称，默认为whisper-small
            base_url (Optional[str]): API基础URL
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)

    async def stt(self, audio: Any, language: str = "zh", prompt: Optional[str] = None, 
                     response_format: str = "json", temperature: float = 0.7, **kwargs) -> tuple[str, int]:
        """
        将音频转录为文本
        
        Args:
            audio: 音频文件对象
            language: 语言代码
            prompt: 提示文本
            response_format: 响应格式
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            tuple: (转录文本, 输入音频的token等价量)
            
        Raises:
            Exception: 当API请求失败时
            
        语言支持方式:
            - 通过language参数指定语言代码
            - 默认"zh"表示中文
            - 支持多种语言代码，如"en"、"ja"、"ko"等
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                if isinstance(audio, str):
                    audio_file = open(audio, "rb")
                    audio_data = audio_file.read()
                    audio_file_name = audio.split("/")[-1]
                    audio_file.close()
                else:
                    audio_data = audio
                    audio_file_name = "audio.wav"

                payload = {
                    "model": self.model_name,
                    "language": language,
                    "prompt": prompt,
                    "response_format": response_format,
                    "temperature": temperature
                }

                files = {"file": (audio_file_name, audio_data, "audio/wav")}

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/v1/audio/transcriptions", 
                        data=payload,
                        files=files
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                if "text" in result:
                    text = result["text"].strip()

                    return text, self._total_token_count(result, audio)
                else:
                    error_text = "**ERROR**: Failed to retrieve transcription."
                    return error_text, 0

            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Xinference STT失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Xinference STT最终失败: {e}")
                    error_text = f"**ERROR**: {str(e)}"
                    return error_text, 0