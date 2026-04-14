from typing import List, Tuple
from urllib.parse import urljoin
import numpy as np
import aiohttp
import asyncio
import logging
from .base import BaseRank, MAX_RETRY_ATTEMPTS
from ..utils import truncate


class XinferenceRank(BaseRank):
    """Xinference重排序模型实现"""
    
    def __init__(self, api_key: str = "x", model_provider: str = "xinference", model_name: str = "", base_url: str = "", **kwargs):
        """
        初始化Xinference重排序模型
        
        Args:
            api_key (str): API密钥，默认为"x"
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL
        """
        if base_url.find("/v1") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        if base_url.find("/rerank") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        self.headers = {
            "Content-Type": "application/json", 
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }      


    async def similarity(self, query: str, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        计算查询与文本列表的相似度分数
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            
        Returns:
            Tuple[np.ndarray, int]: (相似度分数数组, token数量)
        """
        if len(texts) == 0:
            return np.array([]), 0
            
        # 截断文本到4096字符
        texts = [truncate(t, 4096) for t in texts]
            
        data = {
            "model": self.model_name, 
            "query": query, 
            "return_documents": "true", 
            "return_len": "true", 
            "documents": texts
        }
        
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, headers=self.headers, json=data) as response:
                        response.raise_for_status()
                        res = await response.json()
                
                    rank = np.zeros(len(texts), dtype=float)
                    
                    for d in res["results"]:
                        rank[d["index"]] = d["relevance_score"]
                    
                    return rank, self._total_token_count(res, texts)
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Xinference重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Xinference重排序最终失败: {e}")
                    raise RuntimeError(f"**ERROR**: {e}")