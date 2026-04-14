import json
from typing import List, Tuple
import numpy as np
import asyncio
import logging
from qianfan.resources import Reranker
from .base import BaseRank, MAX_RETRY_ATTEMPTS


class BaiduYiyanRank(BaseRank):
    """百度千帆重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str = None, **kwargs):
        """
        初始化百度千帆重排序模型
        
        Args:
            api_key (str): 百度千帆API密钥（JSON格式，包含ak和sk）
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL，可选
        """
        super().__init__(api_key, model_name, base_url, **kwargs)
        
        try:            
            key = json.loads(api_key)
            ak = key.get("yiyan_ak", "")
            sk = key.get("yiyan_sk", "")
            self.client = Reranker(ak=ak, sk=sk)
        except json.JSONDecodeError:
            raise ValueError("API key must be a valid JSON string containing 'yiyan_ak' and 'yiyan_sk'")

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
                    lambda: self.client.do(
                        model=self.model_name,
                        query=query,
                        documents=texts,
                        top_n=len(texts),
                    ).body
                )
                
                rank = np.zeros(len(texts), dtype=float)
                
                for d in res["results"]:
                    rank[d["index"]] = d["relevance_score"]
                    
                return rank, self._total_token_count(res, texts)
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"百度千帆重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"百度千帆重排序最终失败: {e}")
                    raise RuntimeError(f"**ERROR**: {e}")