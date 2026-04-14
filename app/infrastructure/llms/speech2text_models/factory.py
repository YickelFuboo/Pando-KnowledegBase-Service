from typing import Dict, Type
from ..base_factory import BaseModelFactory
from .base import BaseSTT
from .openai_stt import OpenAISTT
from .qwen_stt import QwenSTT
from .azure_stt import AzureSTT
from .tencent_stt import TencentSTT
from .xinference_stt import XinferenceSTT
from .gpustack_stt import GPUStackSTT
from .gitee_stt import GiteeSTT


class STTFactory(BaseModelFactory[BaseSTT]):
    """STT模型工厂类"""
    
    @property
    def _models(self) -> Dict[str, Type[BaseSTT]]:
        return {
            "openai": OpenAISTT,
            "qwen": QwenSTT,
            "azure": AzureSTT,
            "tencent": TencentSTT,
            "xinference": XinferenceSTT,
            "gpustack": GPUStackSTT,
            "gitee": GiteeSTT,
        }

    def __init__(self):
        super().__init__("stt_models.json")


# 全局工厂实例
stt_factory = STTFactory()
