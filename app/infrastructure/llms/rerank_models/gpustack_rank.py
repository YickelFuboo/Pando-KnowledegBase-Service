import logging
from typing import List, Tuple
import httpx
import numpy as np
import aiohttp
import asyncio
from .base import BaseRank, MAX_RETRY_ATTEMPTS


class GPUStackRank(BaseRank):
    """GPUStack重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str, **kwargs):
        """
        初始化GPUStack重排序模型
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL
            **kwargs: 其他参数
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)

        self.base_url = f"{base_url.rstrip('/')}/v1/rerank"

        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
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
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, json=payload, headers=self.headers) as response:
                        response.raise_for_status()
                        response_json = await response.json()

                    rank = np.zeros(len(texts), dtype=float)
                        
                    try:
                        for result in response_json["results"]:
                            rank[result["index"]] = result["relevance_score"]
                    except Exception as e:
                        logging.error(f"GPUStack重排序失败: {e}")

                    return rank, self._total_token_count(response_json, texts)

            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"GPUStack重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"GPUStack重排序最终失败: {e}")
                    raise ValueError(f"Error calling GPUStackRank model {self.model_name}: {e}")