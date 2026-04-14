from typing import Dict, Type
from app.infrastructure.llms.base_factory import BaseModelFactory
from .baai_rank import BAAIRank
from .baidu_yiyan_rank import BaiduYiyanRank
from .base import BaseRank
from .cohere_rank import CohereRank
from .gpustack_rank import GPUStackRank
from .huggingface_rank import HuggingfaceRank
from .nvidia_rank import NvidiaRank
from .openai_rank import OpenAIRank
from .qwen_rank import QwenRank
from .siliconflow_rank import SiliconflowRank
from .voyage_rank import VoyageRank
from .xinference_rank import XinferenceRank


class ReRankFactory(BaseModelFactory[BaseRank]):
    """重排序模型工厂类"""

    @property
    def _models(self) -> Dict[str, Type[BaseRank]]:
        return {
            "baai": BAAIRank,
            "jina": OpenAIRank,
            "xinference": XinferenceRank,
            "cohere": CohereRank,
            "openai_compatible": OpenAIRank,
            "qwen": QwenRank,
            "nvidia": NvidiaRank,
            "voyage": VoyageRank,
            "baidu_yiyan": BaiduYiyanRank,
            "huggingface": HuggingfaceRank,
            "gpustack": GPUStackRank,
            "siliconflow": SiliconflowRank,
        }

    def __init__(self):
        super().__init__("rerank_models.json")


# 全局工厂实例
rerank_factory = ReRankFactory()
