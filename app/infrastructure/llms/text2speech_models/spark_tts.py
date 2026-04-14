import json
import base64
import hashlib
import hmac
import queue
import ssl
import time
import asyncio
import logging
import _thread as thread
from datetime import datetime
from time import mktime
from typing import Generator, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
from .base import BaseTTS, MAX_RETRY_ATTEMPTS


class SparkTTS(BaseTTS):
    """讯飞Spark的文本转语音模型实现"""
    
    STATUS_FIRST_FRAME = 0
    STATUS_CONTINUE_FRAME = 1
    STATUS_LAST_FRAME = 2

    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化Spark TTS模型
        
        Args:
            api_key (str): JSON格式的API密钥，包含spark_app_id、spark_api_secret、spark_api_key
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (Optional[str]): API基础URL（Spark使用WebSocket，此参数忽略）
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        # 解析API密钥
        key_data = json.loads(api_key)
        self.APPID = key_data.get("spark_app_id", "xxxxxxx")
        self.APISecret = key_data.get("spark_api_secret", "xxxxxxx")
        self.APIKey = key_data.get("spark_api_key", "xxxxxx")
        self.CommonArgs = {"app_id": self.APPID}
        self.audio_queue = queue.Queue()

    def create_url(self) -> str:
        """
        生成WebSocket连接URL
        
        Returns:
            str: WebSocket连接URL
        """
        url = "wss://tts-api.xfyun.cn/v2/tts"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding="utf-8")
        
        authorization_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(encoding="utf-8")
        
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        url = url + "?" + urlencode(v)
        return url

    async def tts(self, text: str, **kwargs) -> tuple[Generator[bytes, None, None], int]:
        """
        将文本转换为语音
        
        Args:
            text (str): 待转换的文本
            **kwargs: 其他参数            
                Voice参数说明:
                    - 此模型不支持voice参数
                    - 通过model_name参数指定声音类型
                    - 常用声音：xiaoyan(小燕)、xiaoyu(小雨)、xiaofeng(小枫)等
                    - 声音类型在初始化时通过model_name设置
            
        Returns:
            tuple: (音频数据生成器, 输入文本的token数量)
            
        Raises:
            Exception: 当API请求失败时
        """
        BusinessArgs = {
            "aue": "lame",
            "sfl": 1,
            "auf": "audio/L16;rate=16000",
            "vcn": self.model_name,
            "tte": "utf8"
        }
        Data = {
            "status": 2,
            "text": base64.b64encode(text.encode("utf-8")).decode("utf-8")
        }
        CommonArgs = {"app_id": self.APPID}
        audio_queue = self.audio_queue
        model_name = self.model_name

        class Callback:
            """WebSocket回调处理类"""
            def __init__(self):
                self.audio_queue = audio_queue

            def on_message(self, ws, message):
                """消息回调"""
                message = json.loads(message)
                code = message["code"]
                sid = message["sid"]
                audio = message["data"]["audio"]
                audio = base64.b64decode(audio)
                status = message["data"]["status"]
                
                if status == 2:
                    ws.close()
                if code != 0:
                    errMsg = message["message"]
                    raise Exception(f"sid:{sid} call error:{errMsg} code:{code}")
                else:
                    self.audio_queue.put(audio)

            def on_error(self, ws, error):
                """错误回调"""
                raise Exception(error)

            def on_close(self, ws, close_status_code, close_msg):
                """关闭回调"""
                self.audio_queue.put(None)  # 放入 None 作为结束标志

            def on_open(self, ws):
                """打开回调"""
                def run(*args):
                    d = {"common": CommonArgs, "business": BusinessArgs, "data": Data}
                    ws.send(json.dumps(d))

                thread.start_new_thread(run, ())

        # 计算输入文本的token数量
        input_tokens = self._total_token_count(text) 

        def audio_generator():
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    wsUrl = self.create_url()
                    websocket.enableTrace(False)
                    callback = Callback()
                    ws = websocket.WebSocketApp(
                        wsUrl,
                        on_open=callback.on_open,
                        on_error=callback.on_error,
                        on_close=callback.on_close,
                        on_message=callback.on_message
                    )
                    
                    status_code = 0
                    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                    
                    while True:
                        audio_chunk = self.audio_queue.get()
                        if audio_chunk is None:
                            if status_code == 0:
                                raise Exception(
                                    f"Fail to access model({model_name}) using the provided credentials. "
                                    f"**ERROR**: Invalid APPID, API Secret, or API Key."
                                )
                            else:
                                return  # 成功，退出重试循环
                        status_code = 1
                        yield audio_chunk
                        
                except Exception as e:
                    if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                        delay = self._get_delay(attempt)
                        logging.warning(f"Spark TTS失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                        time.sleep(delay)  # 使用同步sleep，因为这是同步生成器
                        continue
                    else:
                        logging.error(f"Spark TTS最终失败: {e}")
                        raise Exception(f"**ERROR**: {e}")

        return audio_generator(), input_tokens