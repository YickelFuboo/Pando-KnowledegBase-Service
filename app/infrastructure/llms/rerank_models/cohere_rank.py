from typing import List, Tuple
import numpy as np
import asyncio
import logging
from cohere import Client
from .base import BaseRank, MAX_RETRY_ATTEMPTS


class CohereRank(BaseRank):
    """Cohere重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str = None, **kwargs):
        """
        初始化Cohere重排序模型
        
        Args:
            api_key (str): Cohere API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL，可选
        """
        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
        
        self.client = Client(api_key=api_key, base_url=base_url)
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
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                res = await asyncio.to_thread(
                    self.client.rerank,
                    model=self.model_name,
                    query=query,
                    documents=texts,
                    top_n=len(texts),
                    return_documents=False,
                )
                
                rank = np.zeros(len(texts), dtype=float)
                
                for d in res.results:
                    rank[d.index] = d.relevance_score
                
                return rank, self._total_token_count(res, texts)
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"Cohere重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"Cohere重排序最终失败: {e}")
                    raise RuntimeError(f"**ERROR**: {e}")