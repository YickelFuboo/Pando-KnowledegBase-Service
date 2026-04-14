import asyncio
import logging
from typing import Generator, Optional
import dashscope
from .base import BaseTTS, MAX_RETRY_ATTEMPTS
from app.infrastructure.llms.utils import num_tokens_from_string


class QwenTTS(BaseTTS):
    """通义千问文本转语音模型实现（使用dashscope SDK）"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "qwen-tts", base_url: Optional[str] = None, **kwargs):
        """
        初始化通义千问TTS模型
        
        Args:
            api_key (str): 阿里云API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称，默认为qwen-tts
            base_url (Optional[str]): API基础URL（使用dashscope，此参数忽略）
            
        注意：
            - 使用dashscope.audio.qwen_tts.SpeechSynthesizer API
            - 仅支持qwen-tts系列模型
            - 支持多种音色选择：Cherry, Serena, Ethan, Chelsie
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        try:
            dashscope.api_key = api_key
        except ImportError:
            raise ImportError("请安装dashscope库: pip install dashscope")

    async def tts(self, text: str, voice: str = "Cherry", **kwargs) -> tuple[Generator[bytes, None, None], int]:
        """
        将文本转换为语音
        
        Args:
            text (str): 待转换的文本
            voice (str): 音色选择，默认为"Cherry"
            **kwargs: 其他参数
            
        Returns:
            tuple: (音频数据生成器, 输入文本的token数量)
            
        Raises:
            RuntimeError: 当API请求失败时
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # 使用qwen_tts.SpeechSynthesizer API调用方式
                response = await asyncio.to_thread(
                    lambda: dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                        model=self.model_name,
                        api_key=self.api_key,
                        text=text,
                        voice=voice
                    )
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    # 根据官方文档格式获取音频数据
                    audio_url = None
                    audio_data = None
                    
                    # 从response.output.audio.url获取音频URL
                    try:
                        if hasattr(response, 'output') and response.output:
                            output = response.output
                            if hasattr(output, 'audio') and output.audio:
                                audio = output.audio
                                if hasattr(audio, 'url') and audio.url:
                                    audio_url = audio.url
                                elif isinstance(audio, dict) and 'url' in audio:
                                    audio_url = audio['url']
                    except Exception as e:
                        raise RuntimeError(f"解析音频URL失败: {e}")
                    
                    # 如果有URL，下载音频数据
                    if audio_url:
                        import requests
                        try:
                            audio_response = requests.get(audio_url)
                            if audio_response.status_code == 200:
                                audio_data = audio_response.content
                            else:
                                raise RuntimeError(f"下载音频文件失败，状态码: {audio_response.status_code}")
                        except Exception as e:
                            raise RuntimeError(f"下载音频文件失败: {e}")
                    else:
                        raise RuntimeError("响应中没有找到音频URL")
                    
                    # 创建生成器
                    def audio_generator():
                        if audio_data:
                            yield audio_data
                        else:
                            raise RuntimeError("音频数据为空")
                    
                    # 从response.usage.input_tokens获取token数量
                    token_count = 0
                    try:
                        if hasattr(response, 'usage') and response.usage:
                            usage = response.usage
                            if hasattr(usage, 'input_tokens'):
                                token_count = usage.input_tokens
                            elif isinstance(usage, dict) and 'input_tokens' in usage:
                                token_count = usage['input_tokens']
                    except:
                        pass
                    
                    # 如果没有获取到token数量，则计算
                    if token_count == 0:
                        token_count = num_tokens_from_string(text)
                    
                    return audio_generator(), token_count
                else:
                    raise RuntimeError(f"TTS API调用失败: {response.message}")
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Qwen TTS失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Qwen TTS最终失败: {e}")
                    import traceback
                    error_msg = f"TTS转换失败: {e}\nTraceback: {traceback.format_exc()}"
                    raise RuntimeError(error_msg)