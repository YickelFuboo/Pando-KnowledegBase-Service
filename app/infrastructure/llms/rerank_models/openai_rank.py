from typing import List, Tuple
from urllib.parse import urljoin
import numpy as np
import aiohttp
import asyncio
import logging
from .base import BaseRank, MAX_RETRY_ATTEMPTS
from ..utils import truncate


class OpenAIRank(BaseRank):
    """OpenAI兼容的重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str, **kwargs):
        """
        初始化OpenAI兼容的重排序模型
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL
        """
        if base_url.find("/rerank") == -1:
            base_url = urljoin(base_url, "/rerank")
        
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
            
        self.headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {api_key}"
        }
        self.model_name = model_name.split("___")[0]

    async def similarity(self, query: str, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        计算查询与文本列表的相似度分数
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            
        Returns:
            Tuple[np.ndarray, int]: (相似度分数数组, token数量)
        """
        # 截断到 500 tokens（utils.truncate 按 tiktoken 计）
        texts = [truncate(t, 500) for t in texts]
        
        data = {    
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
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
                        
                    # 将rank值标准化到0到1的范围
                    min_rank = np.min(rank)
                    max_rank = np.max(rank)
                    
                    # 避免除零错误
                    if max_rank - min_rank != 0:
                        rank = (rank - min_rank) / (max_rank - min_rank)
                    else:
                        rank = np.zeros_like(rank)
                    
                    return rank, self._total_token_count(res, texts)
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"OpenAI重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"OpenAI重排序最终失败: {e}")
                    raise RuntimeError(f"**ERROR**: {e}")