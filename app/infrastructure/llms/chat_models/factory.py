import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from .base import LLM
from .claude_llm import ClaudeModels
from .openai_llm import OpenAIModels
from .schemes import ChatResponse, TokenUsage
from .zhipu_llm import ZhiPuModels
from ..base_factory import BaseModelFactory

# =============================================================================
# 聊天模型工厂
# =============================================================================

class LLMFactory(BaseModelFactory):
    """聊天模型工厂。``llm_factory.chat``：同时传入 provider 与 model_name 且已启用、未熔断时优先；否则按已启用列表（名称排序）；连续失败达阈值则约 10 分钟内跳过该模型。"""

    @property
    def _models(self) -> Dict[str, Type[LLM]]:
        return {
            "deepseek": OpenAIModels,
            "claude": ClaudeModels,
            "openai": OpenAIModels,
            "qwen": OpenAIModels,
            "siliconflow": OpenAIModels,
            "zhipu": ZhiPuModels,
            "huoshanfangzhou": OpenAIModels,
            "minimax": OpenAIModels,
        }

    def __init__(self) -> None:
        super().__init__("chat_models.json")

llm_factory = LLMFactory()
