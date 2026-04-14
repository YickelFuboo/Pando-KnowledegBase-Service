from typing import List, Tuple
import numpy as np
import aiohttp
import asyncio
import logging
from .base import BaseRank, MAX_RETRY_ATTEMPTS


class HuggingfaceRank(BaseRank):
    """HuggingFace重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str = "BAAI/bge-reranker-v2-m3", base_url: str = "http://127.0.0.1", **kwargs):
        """
        初始化HuggingFace重排序模型
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL
            **kwargs: 其他参数
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        self.model_name = model_name.split("___")[0]

    @staticmethod
    async def post(query: str, texts: List[str], url: str = "127.0.0.1") -> np.ndarray:
        """
        发送POST请求到HuggingFace重排序服务
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            url (str): 服务URL
            
        Returns:
            np.ndarray: 相似度分数数组
        """
        exc = None
        scores = [0 for _ in range(len(texts))]
        batch_size = 8
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(texts), batch_size):
                for attempt in range(MAX_RETRY_ATTEMPTS):
                    try:
                        async with session.post(
                            f"http://{url}/rerank", 
                            headers={"Content-Type": "application/json"}, 
                            json={
                                "query": query, 
                                "texts": texts[i : i + batch_size], 
                                "raw_scores": False, 
                                "truncate": True
                            }
                        ) as response:
                            for o in await response.json():
                                scores[o["index"] + i] = o["score"]
                        break  # 成功，跳出重试循环
                    except Exception as e:
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            delay = 1.0 * (2 ** attempt)  # 简单的指数退避
                            logging.warning(f"HuggingFace重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            exc = e

        if exc:
            raise exc
        return np.array(scores)

    async def similarity(self, query: str, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        计算查询与文本列表的相似度分数
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            
        Returns:
            Tuple[np.ndarray, int]: (相似度分数数组, token数量)
        """            
        return await HuggingfaceRank.post(query, texts, self.base_url), self._total_token_count(None, texts)  