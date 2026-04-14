import logging
import asyncio
from typing import List, Tuple
import numpy as np
import aiohttp
import urllib.parse
from .base import BaseRank, MAX_RETRY_ATTEMPTS


class SiliconflowRank(BaseRank):
    """SILICONFLOW重排序模型实现"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: str = "https://api.siliconflow.cn/v1", **kwargs):
        """
        初始化SILICONFLOW重排序模型
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (str): API基础URL
            **kwargs: 其他参数
        """            
        # 如果 base_url 不包含 "rerank"，则添加
        if not base_url.endswith("rerank"):
            if base_url.endswith("/"):
                base_url = base_url + "rerank"
            else:
                base_url = base_url + "/rerank"

        super().__init__(api_key, model_provider, model_name, base_url, **kwargs)
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
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }
        
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.base_url, json=payload, headers=self.headers) as response:
                        response.raise_for_status()
                        response_json = await response.json()
                    
                    rank = np.zeros(len(texts), dtype=float)
                    try:
                        for d in response_json["results"]:
                            rank[d["index"]] = d["relevance_score"]
                    except Exception as e:
                        logging.error(f"Siliconflow重排序失败: {e}")
                    
                    # token统计
                    token_count = response_json["meta"]["tokens"]["input_tokens"] + response_json["meta"]["tokens"]["output_tokens"]
                        
                    return rank, token_count
                
            except Exception as e:
                if attempt < MAX_RETRY_ATTEMPTS - 1 and self._is_retryable_error(e):
                    delay = self._get_delay(attempt)
                    logging.warning(f"SiliconFlow重排序失败，重试 (尝试 {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. 等待 {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"SiliconFlow重排序最终失败: {e}")
                    raise ValueError(f"Error calling SiliconflowRank model {self.model_name}: {e}")