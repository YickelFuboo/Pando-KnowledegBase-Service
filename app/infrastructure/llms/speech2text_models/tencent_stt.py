import json
import re
import time
import logging
from typing import Any, Optional
import asyncio
from .base import BaseSTT, MAX_RETRY_ATTEMPTS
from tencentcloud.asr.v20190614 import asr_client
from tencentcloud.common import credential
from tencentcloud.asr.v20190614 import models


class TencentSTT(BaseSTT):
    """腾讯云的语音转文本模型实现"""

    def __init__(self, api_key: str, model_provider: str, model_name: str = "16k_zh", base_url: Optional[str] = None, **kwargs):
        """
        初始化腾讯云语音转文本模型
        
        Args:
            api_key (str): 腾讯云API密钥（JSON格式）
            model_provider (str): 模型提供商
            model_name (str): 模型名称，默认为16k_zh
            base_url (Optional[str]): API基础URL，默认为腾讯云官方URL
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        try:

            key_data = json.loads(api_key)
            sid = key_data.get("tencent_cloud_sid", "")
            sk = key_data.get("tencent_cloud_sk", "")
            cred = credential.Credential(sid, sk)
            self.client = asr_client.AsrClient(cred, "")
        except ImportError:
            raise ImportError("请安装腾讯云SDK: pip install tencentcloud-sdk-python")

    async def stt(self, audio: Any, max_retries: int = 60, retry_interval: int = 5, **kwargs) -> tuple[str, int]:
        """
        将音频转录为文本
        
        Args:
            audio: 音频文件对象
            max_retries: 最大重试次数
            retry_interval: 重试间隔（秒）
            **kwargs: 其他参数
            
        Returns:
            tuple: (转录文本, 输入音频的token等价量)
            
        Raises:
            Exception: 当API请求失败时
            
        语言支持方式:
            - 通过model_name参数指定语言
            - 默认"16k_zh"表示中文识别
            - 支持中文、英文等多种模型
            
        接口说明:
            - 使用腾讯云同步识别接口：CreateRecTask + DescribeTaskStatus
            - 适用于短音频（<60秒）
            - 对于长音频，建议使用异步识别接口：CreateAsyncRecognitionTask
        """
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                b64 = self.audio2base64(audio)
                
                # 创建同步识别任务
                req = models.CreateRecTaskRequest()
                params = {
                    "EngineModelType": self.model_name,
                    "ChannelNum": 1,
                    "ResTextFormat": 0,  # 0: 识别结果文本, 1: 词级别时间戳
                    "SourceType": 1,     # 1: 语音数据, 0: 语音文件URL
                    "Data": b64,
                }
                req.from_json_string(json.dumps(params))
                resp = await asyncio.to_thread(
                    lambda: self.client.CreateRecTask(req)
                )

                # 循环查询任务状态
                req = models.DescribeTaskStatusRequest()
                params = {"TaskId": resp.Data.TaskId}
                req.from_json_string(json.dumps(params))
                
                retries = 0
                while retries < max_retries:
                    respone = await asyncio.to_thread(
                        lambda: self.client.DescribeTaskStatus(req)
                    )
                    if respone.Data.StatusStr == "success":
                        # 清理时间戳标记
                        text = re.sub(r"\[\d+:\d+\.\d+,\d+:\d+\.\d+\]\s*", "", respone.Data.Result).strip()
                                         
                        return text, self._total_token_count(respone, audio)  
                    elif respone.Data.StatusStr == "failed":
                        error_text = "**ERROR**: Failed to retrieve speech recognition results."
                        return error_text, 0
                    else:
                        # 任务仍在处理中，等待后重试
                        await asyncio.sleep(retry_interval)
                        retries += 1
                
                error_text = "**ERROR**: Max retries exceeded. Task may still be processing."
                return error_text, 0

            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"腾讯云 STT失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"腾讯云 STT最终失败: {e}")
                    error_text = "**ERROR**: " + str(e)
                    return error_text, 0