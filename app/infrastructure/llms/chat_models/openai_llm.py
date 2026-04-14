import asyncio
import logging
from typing import Any,AsyncGenerator,Dict,List,Literal,Optional,Tuple
from openai import AsyncOpenAI
from .base import LLM, MAX_RETRY_ATTEMPTS, build_llm_httpx_timeout
from .schemes import AskToolResponse,ChatResponse,TokenUsage,ToolInfo
from ..utils import num_tokens_from_string


class OpenAIModels(LLM):
    """OpenAI兼容API的通用实现（适用于OpenAI、DeepSeek、Qwen等）"""
    def __init__(self, api_key: str, model_provider: str, model_name: str, base_url: Optional[str] = None, language: str = "Chinese", **kwargs):
        """初始化OpenAI兼容的聊天模型"""
        if not base_url:
            raise ValueError(f"base_url is required")
        
        super().__init__(api_key, model_provider, model_name, base_url, language, **kwargs)

        # 创建w客户端（使用OpenAI兼容接口）
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=build_llm_httpx_timeout(**kwargs),
            max_retries=0,
        )  
            

    def _format_message(
        self,
        system_prompt: str, 
        user_prompt: str, 
        user_question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """格式化消息为 OpenAI API 所需的格式"""
        try:
            messages = []

            # 添加系统提示信息
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 添加对话历史
            if history:
                messages.extend(self._sanitize_history(history))
 
            # 如果有单独的用户问题信息，则添加用户问题信息
            if user_question:
                user_message = f"{user_prompt}\n{user_question}" if user_prompt else user_question
                messages.append({"role": "user", "content": user_message})

            # 如果messages为空
            if not messages:
                logging.error("Messages are empty")
                raise ValueError("Messages are empty")
        
            return messages
        except Exception as e:
            logging.error(f"Error in _format_openai_message: {e}")
            raise e

    def _sanitize_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗历史消息，避免 assistant.tool_calls 与 tool 结果不配对导致 400。"""
        sanitized: List[Dict[str, Any]] = []
        pending_tool_ids: set[str] = set()

        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if not isinstance(tool_calls, list) or not tool_calls:
                    pending_tool_ids.clear()
                    sanitized.append(msg)
                    continue

                ids = {
                    tc.get("id")
                    for tc in tool_calls
                    if isinstance(tc, dict) and isinstance(tc.get("id"), str) and tc.get("id")
                }
                if not ids:
                    pending_tool_ids.clear()
                    sanitized.append(msg)
                    continue

                pending_tool_ids = set(ids)
                sanitized.append(msg)
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if pending_tool_ids and isinstance(tool_call_id, str) and tool_call_id in pending_tool_ids:
                    pending_tool_ids.remove(tool_call_id)
                    sanitized.append(msg)
                elif not pending_tool_ids:
                    continue
                continue

            if pending_tool_ids and sanitized and sanitized[-1].get("role") == "assistant":
                last = dict(sanitized[-1])
                last.pop("tool_calls", None)
                sanitized[-1] = last
                pending_tool_ids.clear()

            sanitized.append(msg)

        if pending_tool_ids and sanitized and sanitized[-1].get("role") == "assistant":
            last = dict(sanitized[-1])
            last.pop("tool_calls", None)
            sanitized[-1] = last

        return sanitized

    def _extract_usage(self, response)->TokenUsage:
        """从响应中提取 token 使用明细（尽量兼容不同 SDK）。"""
        usage=getattr(response,"usage",None)
        if usage is None:
            return TokenUsage()
        
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read_tokens = getattr(usage, "cache_read_tokens", None)
        cache_write_tokens = getattr(usage, "cache_write_tokens", None)
        if not isinstance(cache_read_tokens, int):
            cache_read_tokens = getattr(usage, "cached_tokens", None)
        if not isinstance(cache_read_tokens, int):
            cache_read_tokens = 0
        if not isinstance(cache_write_tokens, int):
            cache_write_tokens = 0
        if isinstance(total_tokens,int) and total_tokens>0:
            in_tok=prompt_tokens if isinstance(prompt_tokens,int) else (input_tokens if isinstance(input_tokens,int) else 0)
            out_tok=completion_tokens if isinstance(completion_tokens,int) else (output_tokens if isinstance(output_tokens,int) else 0)
            return TokenUsage(input_tokens=in_tok,output_tokens=out_tok,cache_read_tokens=cache_read_tokens,cache_write_tokens=cache_write_tokens,total_tokens=total_tokens)
        if isinstance(prompt_tokens,int) and isinstance(completion_tokens,int):
            total=prompt_tokens+completion_tokens+cache_read_tokens+cache_write_tokens
            return TokenUsage(input_tokens=prompt_tokens,output_tokens=completion_tokens,cache_read_tokens=cache_read_tokens,cache_write_tokens=cache_write_tokens,total_tokens=total)
        if isinstance(input_tokens,int) and isinstance(output_tokens,int):
            total=input_tokens+output_tokens+cache_read_tokens+cache_write_tokens
            return TokenUsage(input_tokens=input_tokens,output_tokens=output_tokens,cache_read_tokens=cache_read_tokens,cache_write_tokens=cache_write_tokens,total_tokens=total)
        return TokenUsage(cache_read_tokens=cache_read_tokens,cache_write_tokens=cache_write_tokens)

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
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                
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
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                
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
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                
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
                response = await self.client.chat.completions.create(model=self.model_name, messages=messages, **params)
                
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
