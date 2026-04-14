import asyncio
import json
import logging
import random
from abc import ABC
from typing import Any,AsyncGenerator,Dict,List,Literal,Optional,Tuple
import httpx
from .schemes import AskToolResponse,ChatResponse,ModelLimits,TokenUsage


# 重试配置常量
MAX_RETRY_ATTEMPTS = 3  # 最大尝试次数
RETRY_DELAY = 2  # 重试间隔（秒）
CONNECTION_TIMEOUT = 30  # 默认连接超时（秒）
DEFAULT_READ_TIMEOUT = 300.0  # 默认读超时（秒），LLM 首包与流式间隔
DEFAULT_WRITE_TIMEOUT = 120.0  # 默认写超时（秒）
DEFAULT_POOL_TIMEOUT = 30.0  # 默认连接池超时（秒）


def build_llm_httpx_timeout(**kwargs: Any) -> httpx.Timeout:
    """从模型配置构造 httpx 分阶段超时，供 openai.AsyncOpenAI 等接受 httpx.Timeout 的客户端使用（DeepSeek/Qwen/SiliconFlow 等走同一 SDK 即复用，与是否 OpenAI 厂商无关）。"""
    def _to_float(key: str, default: float) -> float:
        v = kwargs.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return httpx.Timeout(
        connect=_to_float("timeout_connect", float(CONNECTION_TIMEOUT)),
        read=_to_float("timeout_read", DEFAULT_READ_TIMEOUT),
        write=_to_float("timeout_write", DEFAULT_WRITE_TIMEOUT),
        pool=_to_float("timeout_pool", DEFAULT_POOL_TIMEOUT),
    )


class LLM(ABC):
    """LLM基类，提供通用的聊天功能和工具调用支持"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, language: str = "Chinese", **kwargs):
        """
        初始化LLM基类
        
        Args:
            api_key (str): API密钥
            model_provider (str): 模型提供商
            model_name (str): 模型名称
            base_url (Optional[str]): API基础URL
            language (str): 语言设置，默认为中文
            kwargs (dict): 其他参数
        """
        self.api_key = api_key
        self.model_provider = model_provider
        self.model_name = model_name
        self.base_url = base_url
        self.language = language
        self.configs = kwargs
        self.max_length = kwargs.get("max_length", 8192)
        self.limits = ModelLimits(
            context_limit=(kwargs.get("context_limit") if isinstance(kwargs.get("context_limit"), int) else None),
            max_output_tokens=(kwargs.get("max_tokens") if isinstance(kwargs.get("max_tokens"), int) else None),
            max_input_tokens=(kwargs.get("max_input_tokens") if isinstance(kwargs.get("max_input_tokens"), int) else None),
        )

    def _format_message(
        self,
        system_prompt: str, 
        user_prompt: str, 
        user_question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """格式化消息为 OpenAI API 所需的格式
        
        Args:
            system_prompt (str): 系统提示词
            user_prompt (str): 用户提示词
            user_question (str): 用户问题
            history (Optional[List[Dict[str, Any]]]): 历史消息
        Returns:
            List[Dict[str, Any]]: 格式化后的消息
        """
        pass


    async def chat(self, 
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  with_think: Optional[bool] = False,
                  **kwargs) -> Tuple[ChatResponse, TokenUsage]:
        """执行聊天对话，子类必须实现
        
        Args:
            system_prompt (str): 系统提示词
            user_prompt (str): 用户提示词
            user_question (str): 用户问题
            history (Optional[List[Dict[str, Any]]]): 历史消息
            with_think (bool): 是否开启思考
            kwargs (dict): 其他参数
        Returns:
            ChatResponse: 聊天响应
        """
        pass


    async def chat_stream(self, 
                system_prompt: str,
                user_prompt: str,
                user_question: str,
                history: List[Dict[str, Any]] = None,
                with_think: Optional[bool] = False,
                **kwargs) -> Tuple[AsyncGenerator[str, None], TokenUsage]:
        """执行聊天对话，子类必须实现
        
        Args:
            system_prompt (str): 系统提示词
            user_prompt (str): 用户提示词
            user_question (str): 用户问题
            history (Optional[List[Dict[str, Any]]]): 历史消息
            with_think (bool): 是否开启思考
            kwargs (dict): 其他参数
        Returns:
            ChatResponse: 聊天响应
        """
        pass


    async def ask_tools(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, Any]] = None,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       with_think: Optional[bool] = False,
                       **kwargs) -> Tuple[AskToolResponse, TokenUsage]:
        """执行工具调用，子类必须实现
        
        Args:
            system_prompt (str): 系统提示词
            user_prompt (str): 用户提示词
            user_question (str): 用户问题
            history (Optional[List[Dict[str, Any]]]): 历史消息
            tools (Optional[List[dict]]): 工具列表
            tool_choice (Literal["none", "auto", "required"]): 工具选择模式
            with_think (bool): 是否开启思考
            kwargs (dict): 其他参数
        Returns:
            AskToolResponse: 工具调用响应
        """
        pass

    async def ask_tools_stream(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, Any]] = None,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       with_think: Optional[bool] = False,
                       **kwargs) -> Tuple[AsyncGenerator[str, None], TokenUsage]:
        """执行工具调用，子类必须实现
        
        Args:
            system_prompt (str): 系统提示词
            user_prompt (str): 用户提示词
            user_question (str): 用户问题
            history (Optional[List[Dict[str, Any]]]): 历史消息
            tools (Optional[List[dict]]): 工具列表
            tool_choice (Literal["none", "auto", "required"]): 工具选择模式
            with_think (bool): 是否开启思考
            kwargs (dict): 其他参数
        Returns:
            AskToolResponse: 工具调用响应
        """
        pass
    
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试）"""
        if self._is_context_overflow_error(error):
            return False
        
        error_str = str(error).lower()        
        # 扩展重试条件，包含更多网络相关错误
        retryable_keywords = [
            'rate limit', '429', 'server', '502', '503', '504', '500',
            'connection', 'timeout', 'timed out', 'network', 'temporary', 'busy', 
            'overload', 'service unavailable', 'internal server error',
            'bad gateway', 'gateway timeout', 'too many requests'
        ]
        
        return any(keyword in error_str for keyword in retryable_keywords)

    def _is_context_overflow_error(self, error: Exception) -> bool:
        s = str(error).lower()
        keywords = (
            "context length",
            "maximum context",
            "max context",
            "context window",
            "exceeds the context",
            "exceed context",
            "too many tokens",
            "prompt is too long",
            "input is too long",
            "context_limit",
        )
        return any(k in s for k in keywords)


    def _get_delay(self, attempt: int = 0):
        """获取重试延迟时间（指数退避 + 随机抖动）"""
        # 指数退避：2^attempt * 基础延迟
        base_delay = 1.0
        exponential_delay = base_delay * (2 ** attempt)
        
        # 添加随机抖动，避免雷群效应
        jitter = random.uniform(0.5, 1.5)
        
        # 限制最大延迟为30秒
        max_delay = 30.0
        delay = min(exponential_delay * jitter, max_delay)
        
        return delay

    
    def _create_error_stream(self, error_message: str):
        """创建错误流"""
        async def error_stream():
            yield str(error_message)
        return error_stream()
    

    def _add_truncate_notify(self, content: str) -> str:
        """
        截断响应文本并添加截断提示
        
        Args:
            content (str): 响应文本            
        Returns:
            str: 截断后的文本
        """
        if self.language.lower() == "chinese":
            content += "······\n由于大模型的上下文窗口大小限制，回答已经被大模型截断。"
        else:
            content += "...\nThe answer is truncated by your chosen LLM due to its limitation on context length."
        
        return content

    def _calculate_dynamic_ctx(self, history: List[Dict[str, Any]]):
        """计算动态上下文窗口大小"""
        def _count_tokens(text: str) -> int:
            # 简单计算：ASCII字符1个token，非ASCII字符（中文、日文、韩文等）2个token
            total = 0
            for char in text:
                if ord(char) < 128:  # ASCII字符
                    total += 1
                else:
                    total += 2  # 非ASCII字符（中文、日文、韩文等）
            return total

        total_tokens = 0
        for message in history:
            # 计算内容token数
            content = message.get("content", "")
            content_tokens = _count_tokens(content)
            # 添加角色标记token开销
            role_tokens = 4
            total_tokens += content_tokens + role_tokens

        # 应用1.2倍缓冲比率
        total_tokens_with_buffer = int(total_tokens * 1.2)

        if total_tokens_with_buffer <= 8192:
            ctx_size = 8192
        else:
            ctx_multiplier = (total_tokens_with_buffer // 8192) + 1
            ctx_size = ctx_multiplier * 8192

        return ctx_size

    def _format_tool_calls(self, tool_calls_data: dict) -> str:
        """
        格式化多个工具调用信息为字符串
        
        Args:
            tool_calls_data (dict): 工具调用数据，格式为 {tool_id: {id, name, arguments}}
            
        Returns:
            str: 格式化的工具调用字符串
        """
        if not tool_calls_data:
            return ""
        
        result = "<tool_calls>\n"
        for tool_id, tool_data in tool_calls_data.items():
            # 解析参数
            arguments_str = tool_data.get("arguments", "")
            try:
                args = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                args = arguments_str
            
            tool_info = {
                "id": tool_data["id"],
                "name": tool_data["name"],
                "args": args
            }
            result += "<tool>\n" + json.dumps(tool_info, ensure_ascii=False, indent=2) + "\n</tool>\n"
        result += "</tool_calls>"
        return result

    async def is_strong_enough(self):
        """
        检查当前聊天模型是否足够强大，通过压力测试验证模型能力
            
        Returns:
            bool: 模型是否足够强大
        """
        async def _is_strong_enough():
            try:
                res, _ = await asyncio.wait_for(
                    self.chat("Nothing special.", "", "Are you strong enough!?", [{"role":"user", "content": "Are you strong enough!?"}]),
                    timeout=30
                )
                if not res.success or res.content.find("**ERROR**") >= 0:
                    raise Exception(res.content)
            except asyncio.TimeoutError:
                raise Exception("Chat model timeout")

        # Pressure test for GraphRAG task
        try:
            tasks = [_is_strong_enough() for _ in range(32)]
            await asyncio.gather(*tasks, return_exceptions=True)
            return True
        except Exception as e:
            logging.error(f"聊天模型强度测试失败: {e}")
            return False