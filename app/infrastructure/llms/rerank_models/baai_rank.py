import os
import logging
import re
import threading
from collections.abc import Iterable
from typing import List, Tuple, Optional
import numpy as np
import asyncio
import torch
from FlagEmbedding import FlagReranker
from huggingface_hub import snapshot_download
from .base import BaseRank
from ..utils import truncate


class BAAIRank(BaseRank):
    """BAAI重排序模型实现，使用FlagReranker"""
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, api_key: str, model_provider: str, model_name: str, **kwargs):
        """
        初始化BAAI重排序模型
        
        Args:
            api_key (str): API密钥（未使用）
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            **kwargs: 其他参数
            
        注意：
        如果下载HuggingFace模型遇到问题，可以尝试以下方法：
        
        Linux:
        export HF_ENDPOINT=https://hf-mirror.com
        """
        super().__init__(api_key, model_provider, model_name, **kwargs)
        
        # 加载BAAI FlagReranker模型
        if not BAAIRank._model:
            with BAAIRank._model_lock:
                if not BAAIRank._model:
                    try:
                        # 使用基类的缓存路径获取方法
                        model_path = BaseRank._get_model_cache_path(model_name)
                        
                        # 尝试直接加载本地模型
                        if os.path.exists(model_path):
                            BAAIRank._model = FlagReranker(model_path, use_fp16=torch.cuda.is_available())
                        else:
                            # 从HuggingFace下载模型
                            model_dir = snapshot_download(
                                repo_id=model_name, 
                                local_dir=model_path, 
                                local_dir_use_symlinks=False
                            )
                            BAAIRank._model = FlagReranker(model_dir, use_fp16=torch.cuda.is_available())
                            
                    except Exception as e:
                        raise RuntimeError(f"Failed to load BAAI FlagReranker model {model_name}: {e}")
        
        self._model = BAAIRank._model
        self._dynamic_batch_size = 8
        self._min_batch_size = 1

    def torch_empty_cache(self):
        """
        清空CUDA缓存
        """
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"清空CUDA缓存失败: {e}")
    
    def _compute_batch_scores(self, batch_pairs: List[Tuple[str, str]], max_length: Optional[int] = None) -> List[float]:
        """
        计算批次分数
        
        Args:
            batch_pairs: 批次查询-文档对
            max_length: 最大长度
            
        Returns:
            List[float]: 分数列表
        """
        if max_length is None:
            scores = self._model.compute_score(batch_pairs, normalize=True)
        else:
            scores = self._model.compute_score(batch_pairs, max_length=max_length, normalize=True)
            
        if not isinstance(scores, Iterable):
            scores = [scores]
        return scores

    def _process_batch(self, pairs: List[Tuple[str, str]], max_batch_size: Optional[int] = None) -> np.ndarray:
        """
        处理批次数据的模板方法
        
        Args:
            pairs: 查询-文档对列表
            max_batch_size: 最大批次大小
            
        Returns:
            numpy.ndarray: 相似度分数数组
        """
        old_dynamic_batch_size = self._dynamic_batch_size
        if max_batch_size is not None:
            self._dynamic_batch_size = max_batch_size
            
        res = np.array([], dtype=float)
        i = 0
        
        while i < len(pairs):
            cur_i = i
            current_batch = self._dynamic_batch_size
            max_retries = 5
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 调用子类实现的批次处理计算
                    batch_scores = self._compute_batch_scores(pairs[i : i + current_batch])
                    res = np.append(res, batch_scores)
                    i += current_batch
                    self._dynamic_batch_size = min(self._dynamic_batch_size * 2, 8)
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and current_batch > self._min_batch_size:
                        current_batch = max(current_batch // 2, self._min_batch_size)
                        self.torch_empty_cache()
                        i = cur_i  # 重置i到当前批次的开始
                        retry_count += 1
                    else:
                        raise
                        
            if retry_count >= max_retries:
                raise RuntimeError("max retry times, still cannot process batch, please check your GPU memory")
            self.torch_empty_cache()

        self._dynamic_batch_size = old_dynamic_batch_size
        return np.array(res)

    async def similarity(self, query: str, texts: List[str]) -> Tuple[np.ndarray, int]:
        """
        计算查询与文本列表的相似度分数
        
        Args:
            query (str): 查询文本
            texts (List[str]): 待排序的文本列表
            
        Returns:
            Tuple[np.ndarray, int]: (相似度分数数组, token数量)
        """
        if not self._model:
            raise NotImplementedError("Model not loaded. Please install required dependencies.")
            
        # 截断文本到2048字符
        texts = [truncate(t, 2048) for t in texts]
        pairs = [(query, t) for t in texts]
        
        batch_size = 4096
        res = await asyncio.to_thread(
            self._process_batch,
            pairs,
            max_batch_size=batch_size
        )
        
        return np.array(res), self._total_token_count(None, texts)