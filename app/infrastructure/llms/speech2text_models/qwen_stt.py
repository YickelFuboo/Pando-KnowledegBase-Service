import os
from typing import Any, Optional, Tuple
import asyncio
import logging
from .base import BaseSTT, MAX_RETRY_ATTEMPTS
import dashscope
from http import HTTPStatus
from dashscope import MultiModalConversation


class QwenSTT(BaseSTT):
    """通义千问语音识别模型实现（使用dashscope SDK）"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "qwen-audio-asr", base_url: Optional[str] = None, **kwargs):
        """
        初始化通义千问语音识别模型
        
        Args:
            api_key (str): 阿里云API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称，默认为qwen-audio-asr
            base_url (Optional[str]): API基础URL（使用dashscope，此参数忽略）
            
        注意：
            - 使用通义千问ASR模型进行语音识别
            - 支持中文和英文识别
            - 支持最长3分钟的音频
            - 使用dashscope原生API，不支持OpenAI兼容模式
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        dashscope.api_key = api_key

    async def stt(self, audio: Any, format: str = "wav", **kwargs) -> Tuple[str, int]:
        """
        将音频转录为文本
        
        Args:
            audio: 音频输入，支持以下格式：
                - 音频文件URL (str) - 推荐方式，需要公网可访问
                - 文件路径 (str) - 本地文件路径，需要添加file://前缀
                - BytesIO对象 - 音频流（需要保存为临时文件）
                - 字节数据 (bytes) - 原始音频数据（需要保存为临时文件）
            format: 音频格式 (wav, mp3, flac等)
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, int]: (转录文本, token数量)
            
        Raises:
            Exception: 当API请求失败时
            
        注意：
            - 使用MultiModalConversation API调用通义千问ASR模型
            - 支持多种音频格式：wav, mp3, flac, aac等
            - 支持中文和英文识别，最长3分钟音频
            - 通义千问ASR模型专门用于语音识别，准确率较高
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # 处理不同类型的音频输入，转换为文件路径
                audio_file_path = self._prepare_audio_input(audio)
                
                # 构建消息
                messages = [
                    {
                        "role": "user",
                        "content": [{"audio": audio_file_path}],
                    }
                ]
                
                # 调用MultiModalConversation API
                response = await asyncio.to_thread(
                    lambda: MultiModalConversation.call(
                        model=self.model_name, 
                        messages=messages,
                        result_format="message"
                    )
                )

                if response.status_code == HTTPStatus.OK:
                    # 提取转录文本
                    text = ""
                    if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                        for choice in response.output.choices:
                            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                text += str(choice.message.content) + "\n"
                    
                    if not text:
                        # 如果没有找到标准格式，尝试其他可能的字段
                        if hasattr(response, 'output'):
                            text = str(response.output)
                    
                    # 获取token数量
                    return text.strip(), self._total_token_count(response, audio_file_path)

                # 处理错误响应
                error_msg = "Unknown error"
                if hasattr(response, 'message'):
                    error_msg = str(response.message)
                elif hasattr(response, 'output') and hasattr(response.output, 'message'):
                    error_msg = str(response.output.message)
                
                error_text = f"**ERROR**: {error_msg}"
                return error_text, 0
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Qwen STT失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Qwen STT最终失败: {e}")
                    import traceback
                    error_text = f"**ERROR**: {e}\nTraceback: {traceback.format_exc()}"
                    return error_text, 0
    
    def _prepare_audio_input(self, audio: Any) -> str:
        """
        预处理音频输入，转换为MultiModalConversation API支持的格式
        
        Args:
            audio: 原始音频输入，支持：
                - 音频文件URL (str) - 格式: https://...
                - 本地文件路径 (str) - 格式: /path/to/file.wav
            
        Returns:
            str: 音频文件路径，格式为：
                - URL: https://...
                - 本地文件: file://绝对路径
                
        Raises:
            ValueError: 当输入格式不支持时
        """
        # 如果是字符串，可能是URL或文件路径
        if isinstance(audio, str):
            # 检查是否是URL
            if audio.startswith(('http://', 'https://')):
                return audio  # 直接返回URL
            else:
                # 文件路径，检查文件是否存在
                if os.path.exists(audio):
                    # 转换为绝对路径并添加file://前缀
                    abs_path = os.path.abspath(audio)
                    return f"file://{abs_path}"
                else:
                    raise FileNotFoundError(f"音频文件不存在: {audio}")
        
        # 其他格式不支持
        else:
            raise ValueError(f"通义千问ASR模型只支持URL和本地文件路径，不支持 {type(audio)} 类型")