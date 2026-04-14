from typing import Any,AsyncGenerator,Dict,List,Literal,Optional,Tuple
import asyncio
import logging
from zai import ZhipuAiClient
from .base import MAX_RETRY_ATTEMPTS
from .schemes import AskToolResponse,ChatResponse,TokenUsage,ToolInfo
from .openai_llm import OpenAIModels
from ..utils import num_tokens_from_string


class ZhiPuModels(OpenAIModels):
    """智谱AI模型系列"""
    
    def __init__(self, api_key: str, model_provider: str, model_name: str = "glm-5", base_url: Optional[str] = None, language: str = "Chinese", **kwargs):
        """
        初始化智谱AI模型
        
        Args:
            api_key (str): 智谱AI API密钥
            model_name (str): 模型名称，默认为glm-5
            base_url (str): API基础URL，默认为智谱AI官方API
            language (str): 语言设置
            **kwargs: 其他参数
        """
        super().__init__(api_key, model_provider, model_name, base_url, language, **kwargs)
        
        # 创建智谱AI客户端
        self.client = ZhipuAiClient(
            api_key=api_key,
        )

    async def chat(self, 
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  with_think: Optional[bool] = False,
                  **kwargs) -> Tuple[ChatResponse, TokenUsage]:
        """OpenAI兼容的聊天实现，支持失败重试"""
        messages = self._format_message(
            system_prompt, user_prompt, user_question, history
        )

        # 构建参数
        params = {"stream": False}
        # 添加其他参数，避免重复
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        # 实现重试策略
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    **params
                )
                
                # 检查响应结构是否有效
                if (not response.choices or not response.choices[0].message or  not response.choices[0].message.content):
                    return ChatResponse(content="Invalid response structure",success=False),TokenUsage()
                
                # 获取回答内容
                # ps：非流式场景下，即便开启了reasoning_mode: "deep"，也不会返回reasoning_content字段，所有内容
                # （思考 + 答案）合并到Content中返回     
                content = response.choices[0].message.content.strip()
                
                # 检查是否因长度限制截断
                if response.choices[0].finish_reason == "length":
                    content = self._add_truncate_notify(content)
                usage=self._extract_usage(response)
                return ChatResponse(content=content,success=True), usage
            
            except Exception as e:
                if self._is_context_overflow_error(e):
                    logging.error(f"Error in chat (context overflow): {e}")
                    return ChatResponse(content="llm error: context_overflow", success=False), TokenUsage()
                # 检查是否需要重试
                if not self._is_retryable_error(e) or attempt == MAX_RETRY_ATTEMPTS - 1:
                    logging.error(f"Error in chat (attempt {attempt + 1}): {e}")
                    return ChatResponse(content="llm error: " + str(e),success=False),TokenUsage()
                
                # 重试延迟（指数退避）
                delay = self._get_delay(attempt)
                logging.warning(f"Retryable error in chat (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        return ChatResponse(content="llm error: Unexpected error: max retries exceeded",success=False),TokenUsage()

    
    async def chat_stream(self, 
                  system_prompt: str,
                  user_prompt: str,
                  user_question: str,
                  history: List[Dict[str, Any]] = None,
                  with_think: Optional[bool] = False,
                  **kwargs) -> Tuple[AsyncGenerator[str, None], TokenUsage]:
        """OpenAI兼容的聊天流式实现，支持失败重试"""
        messages = self._format_message(
            system_prompt, user_prompt, user_question, history
        )

        # 构建参数
        params = {"stream": True}
        # 添加其他参数，避免重复
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 实现重试策略
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # 调用模型接口
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    **params
                )
                
                # 检查响应结构是否有效
                if not response:
                    return self._create_error_stream("Invalid response structure"), TokenUsage()
                
                usage = TokenUsage()
                
                async def stream_response():
                    nonlocal usage
                    reasoning_start = False  
                    
                    try:
                        async for chunk in response:
                            content = ""

                            # 获取内容
                            if not chunk.choices:
                                continue
                            
                            # 拼接think部分，开启"reasoning_mode": "deep"后有本内容
                            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content is not None:
                                if not reasoning_start:
                                    reasoning_start = True
                                    content = "<think>"
                                content += chunk.choices[0].delta.reasoning_content
                            
                            # 正式内容拼接
                            if chunk.choices[0].delta.content:
                                if reasoning_start:
                                    content += "</think>"
                                    reasoning_start = False
                                content += chunk.choices[0].delta.content 

                            if content:
                                usage.total_tokens += num_tokens_from_string(content)

                            # 如果超长截断，则添加截断提示
                            if chunk.choices[0].finish_reason == "length":
                                content = self._add_truncate_notify(content)

                            yield content

                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                
                # 返回流式响应和token数量
                return stream_response(), usage

            except Exception as e:
                if self._is_context_overflow_error(e):
                    logging.error(f"Error in chat_stream (context overflow): {e}")
                    return self._create_error_stream("llm error: context_overflow"), TokenUsage()
                # 检查是否需要重试
                if not self._is_retryable_error(e) or attempt == MAX_RETRY_ATTEMPTS - 1:
                    logging.error(f"Error in chat_stream (attempt {attempt + 1}): {e}")
                    return self._create_error_stream(str(e)), TokenUsage()
                
                # 重试延迟（指数退避）
                delay = self._get_delay(attempt)
                logging.warning(f"Retryable error in chat_stream (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        return self._create_error_stream("Unexpected error: max retries exceeded"), TokenUsage()


    async def ask_tools(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, Any]] = None,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       with_think: Optional[bool] = False,
                       **kwargs) -> Tuple[AskToolResponse, TokenUsage]:
        """OpenAI兼容的工具调用实现，支持失败重试"""
        if tool_choice == "required" and not tools:
            return AskToolResponse(
                content="tool_choice 为 'required' 时必须提供 tools",
                success=False
            ),TokenUsage()

        messages = self._format_message(
            system_prompt, user_prompt, user_question, history
        )
        
        params = {"stream": False}
        if tools and tool_choice != "none":
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # 添加其他参数，避免重复
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 实现重试策略
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    **params
                )
                
                # 检查响应结构是否有效
                if (not response.choices or not response.choices[0].message):
                    return AskToolResponse(content="llm error: Invalid response structure",success=False),TokenUsage()
                
                msg = response.choices[0].message
                tool_calls = []
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls.append(ToolInfo(
                            id=tool_call.id or "",
                            name=tool_call.function.name or "",
                            args=tool_call.function.arguments or "",
                        ))
                
                usage=self._extract_usage(response)
                return AskToolResponse(content=msg.content or "",tool_calls=tool_calls,success=True), usage

            except Exception as e:
                if self._is_context_overflow_error(e):
                    logging.error(f"Error in ask_tools (context overflow): {e}")
                    return AskToolResponse(content="llm error: context_overflow", success=False), TokenUsage()
                # 检查是否需要重试
                if not self._is_retryable_error(e) or attempt == MAX_RETRY_ATTEMPTS - 1:
                    logging.error(f"Error in ask_tools (attempt {attempt + 1}): {e}")
                    return AskToolResponse(content="llm error: " + str(e),success=False),TokenUsage()
                
                # 重试延迟（指数退避）
                delay = self._get_delay(attempt)
                logging.warning(f"Retryable error in ask_tools (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        return AskToolResponse(content="llm error: Unexpected error: max retries exceeded",success=False),TokenUsage()


    async def ask_tools_stream(self,
                       system_prompt: str,
                       user_prompt: str,
                       user_question: str,
                       history: List[Dict[str, Any]] = None,
                       tools: Optional[List[dict]] = None,
                       tool_choice: Literal["none", "auto", "required"] = "auto",
                       with_think: Optional[bool] = False,
                       **kwargs) -> Tuple[AsyncGenerator[str, None], TokenUsage]:
        """OpenAI兼容的工具调用流式实现，支持失败重试"""
        if tool_choice == "required" and not tools:
            return self._create_error_stream("llm error: tool_choice 为 'required' 时必须提供 tools"), TokenUsage()
        
        messages = self._format_message(
            system_prompt, user_prompt, user_question, history
        )
        
        params = {"stream": True}
        if tools and tool_choice != "none":
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        # 添加其他参数，避免重复
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        # 实现重试策略
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    **params
                )
                
                # 检查响应结构是否有效
                if not response:
                    return self._create_error_stream("llm error: Invalid response structure"), TokenUsage()
                
                usage = TokenUsage()
                
                async def stream_response():
                    nonlocal usage
                    reasoning_start = False
                    tool_calls_collected = {}
                    
                    try:
                        async for chunk in response:
                            content = ""
                   
                            if not chunk.choices:
                                continue
                            
                            if content:
                                usage.total_tokens += num_tokens_from_string(content)

                            # 拼接think部分，开启"reasoning_mode": "deep"后有本内容
                            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content is not None:
                                if not reasoning_start:
                                    reasoning_start = True
                                    content = "<think>"
                                content += chunk.choices[0].delta.reasoning_content

                            # 正式内容拼接
                            if chunk.choices[0].delta.content:
                                if reasoning_start:
                                    content += "</think>"
                                    reasoning_start = False
                                content += chunk.choices[0].delta.content
                            
                            # 处理工具调用
                            if chunk.choices[0].delta.tool_calls:
                                for tc in chunk.choices[0].delta.tool_calls:
                                    if not tc:
                                        continue
                                    idx = getattr(tc, "index", None)
                                    if idx is None:
                                        idx = 0
                                    if idx not in tool_calls_collected:
                                        tool_calls_collected[idx] = {
                                            "id": "",
                                            "name": "",
                                            "arguments": ""
                                        }
                                    item = tool_calls_collected[idx]
                                    if getattr(tc, "id", None):
                                        item["id"] = tc.id
                                    fn = getattr(tc, "function", None)
                                    if fn:
                                        if getattr(fn, "name", None):
                                            item["name"] = fn.name
                                        args_piece = getattr(fn, "arguments", None)
                                        if args_piece:
                                            item["arguments"] += args_piece
                            
                            
                            # 如果有内容则yield（实时返回）
                            if content:
                                yield content

                        # 处理收集到的工具调用，格式化为字符串
                        if tool_calls_collected:
                            ordered = {}
                            for _, item in sorted(tool_calls_collected.items(), key=lambda kv: kv[0]):
                                tool_id = item.get("id") or ""
                                if not tool_id:
                                    tool_id = f"toolcall_{len(ordered)}"
                                ordered[tool_id] = item
                            tool_calls_str = self._format_tool_calls(ordered)
                            usage.total_tokens += num_tokens_from_string(tool_calls_str)
                            yield tool_calls_str
                    
                    except Exception as e:
                        logging.error(f"Error in stream response: {e}")
                        if hasattr(response, 'close'):
                            await response.close()
                        raise
                
                # 返回流式响应和token数量
                return stream_response(), usage

            except Exception as e:
                if self._is_context_overflow_error(e):
                    logging.error(f"Error in ask_tools_stream (context overflow): {e}")
                    return self._create_error_stream("llm error: context_overflow"), TokenUsage()
                # 检查是否需要重试
                if not self._is_retryable_error(e) or attempt == MAX_RETRY_ATTEMPTS - 1:
                    logging.error(f"Error in ask_tools_stream (attempt {attempt + 1}): {e}")
                    return self._create_error_stream("llm error: " + str(e)), TokenUsage()
                
                # 重试延迟（指数退避）
                delay = self._get_delay(attempt)
                logging.warning(f"Retryable error in ask_tools_stream (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}): {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
        
        return self._create_error_stream("llm error: Unexpected error: max retries exceeded"), TokenUsage()