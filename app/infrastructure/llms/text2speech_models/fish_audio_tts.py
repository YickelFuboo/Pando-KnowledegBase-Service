import json
import aiohttp
import asyncio
import logging
import ormsgpack
from typing import Generator, Optional, Literal
from pydantic import BaseModel, conint
from http import HTTPStatus
from .base import BaseTTS, MAX_RETRY_ATTEMPTS


class ServeReferenceAudio(BaseModel):
    """参考音频模型"""
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    """Fish Audio TTS请求模型"""
    text: str
    chunk_length: conint(ge=100, le=300, strict=True) = 200
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    references: list[ServeReferenceAudio] = []
    reference_id: Optional[str] = None
    normalize: bool = True
    latency: Literal["normal", "balanced"] = "normal"


class FishAudioTTS(BaseTTS):
    """Fish Audio的文本转语音模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化Fish Audio TTS模型
        
        Args:
            api_key (str): JSON格式的API密钥，包含fish_audio_ak和fish_audio_refid
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (Optional[str]): API基础URL，默认为Fish Audio官方URL
        """
        if not base_url:
            base_url = "https://api.fish.audio/v1/tts"
        
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        # 解析API密钥
        key_data = json.loads(api_key)
        self.headers = {
            "api-key": key_data.get("fish_audio_ak"),
            "content-type": "application/msgpack",
        }
        self.ref_id = key_data.get("fish_audio_refid")

    async def tts(self, text: str, **kwargs) -> tuple[Generator[bytes, None, None], int]:
        """
        将文本转换为语音
        
        Args:
            text (str): 待转换的文本
            **kwargs: 其他参数
            
        Returns:
            tuple: (音频数据生成器, 输入文本的token数量)
            
        Raises:
            RuntimeError: 当API请求失败时
            
        Voice参数说明:
            - 此模型不支持voice参数
            - 通过reference_id参数指定参考音频
            - 使用参考音频克隆声音特征
            - 支持多种语言，取决于参考音频的语言
            - 声音类型在初始化时通过API密钥中的fish_audio_refid设置
        """
        text = self.normalize_text(text)
        request = ServeTTSRequest(text=text, reference_id=self.ref_id)

        # 计算输入文本的token数量
        input_tokens = self._total_token_count(text)

        async def audio_generator():
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url=self.base_url,
                            data=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                            headers=self.headers,
                            timeout=None,
                            stream=True
                        ) as response:
                            if response.status == HTTPStatus.OK:
                                async for chunk in response.content.iter_chunked(1024):
                                    yield chunk
                                return  # 成功，退出重试循环
                            else:
                                response.raise_for_status()

                except Exception as e:
                    if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                        delay = self._get_delay(attempt)
                        logging.warning(f"Fish Audio TTS失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logging.error(f"Fish Audio TTS最终失败: {e}")
                        raise RuntimeError(f"**ERROR**: {e}")

        return audio_generator(), input_tokens