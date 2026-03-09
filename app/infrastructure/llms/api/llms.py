from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import asyncio
import io
import base64
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.infrastructure.llms import llm_factory, cv_factory, embedding_factory, rerank_factory, stt_factory, tts_factory


# 主路由
router = APIRouter(prefix="/models", tags=["模型管理"])


# ==================== 数据模型 ====================

class ModelRequest(BaseModel):
    """通用模型请求"""
    model_config = {"protected_namespaces": ()}
    
    provider: Optional[str] = None
    model_name: Optional[str] = None


class ChatRequest(ModelRequest):
    """聊天请求"""
    system_prompt: Optional[str] = None
    user_prompt: str
    user_question: str


class ChatResponse(BaseModel):
    """聊天响应"""
    content: str
    token_count: int


class ImageDescribeRequest(ModelRequest):
    """图像描述请求"""
    image_base64: str


class ImageDescribeWithPromptRequest(ModelRequest):
    """带提示词的图像描述请求"""
    image_base64: str
    prompt: str


class ImageChatRequest(ModelRequest):
    """图像聊天请求（支持普通和流式）"""
    image_base64: str
    user_question: str


class EmbeddingRequest(ModelRequest):
    """嵌入请求"""
    texts: List[str]


class EmbeddingResponse(BaseModel):
    """嵌入响应"""
    embeddings: List[List[float]]
    token_count: int


class RerankRequest(ModelRequest):
    """重排序请求"""
    query: str
    texts: List[str]


class RerankResponse(BaseModel):
    """重排序响应"""
    similarities: List[float]


class TTSRequest(ModelRequest):
    """文本转语音请求"""
    text: str
    voice: Optional[str] = None


# ==================== 主路由 - 模型列表查询 ====================

@router.get("/", summary="获取所有支持的模型列表")
async def get_all_models():
    """获取所有功能模块支持的模型列表"""
    try:
        # 直接使用工厂方法获取模型列表
        return {
            "chat_models": llm_factory.get_supported_models(),
            "cv_models": cv_factory.get_supported_models(),
            "embedding_models": embedding_factory.get_supported_models(),
            "rerank_models": rerank_factory.get_supported_models(),
            "stt_models": stt_factory.get_supported_models(),
            "tts_models": tts_factory.get_supported_models()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/available/chat", summary="获取可用聊天模型列表")
async def get_chat_models():
    """获取可用聊天模型列表"""
    return llm_factory.get_supported_models()


@router.get("/available/cv", summary="获取可用计算机视觉模型列表")
async def get_cv_models():
    """获取可用计算机视觉模型列表"""
    return cv_factory.get_supported_models()


@router.get("/available/embedding", summary="获取可用嵌入模型列表")
async def get_embedding_models():
    """获取可用嵌入模型列表"""
    return embedding_factory.get_supported_models()


@router.get("/available/rerank", summary="获取可用重排序模型列表")
async def get_rerank_models():
    """获取可用重排序模型列表"""
    return rerank_factory.get_supported_models()


@router.get("/available/stt", summary="获取可用语音转文本模型列表")
async def get_stt_models():
    """获取可用语音转文本模型列表"""
    return stt_factory.get_supported_models()


@router.get("/available/tts", summary="获取可用文本转语音模型列表")
async def get_tts_models():
    """获取可用文本转语音模型列表"""
    return tts_factory.get_supported_models()


# ==================== 聊天模型API ====================

@router.post("/chat", response_model=ChatResponse, summary="聊天对话", tags=["聊天模型"])
async def chat(request: ChatRequest):
    """聊天对话接口"""
    try:
        model = llm_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        response, token_count = await model.chat(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            user_question=request.user_question
        )
        
        return ChatResponse(
            content=response.content if hasattr(response, 'content') else str(response),
            token_count=token_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天请求失败: {str(e)}")


@router.post("/chat/stream", summary="流式聊天对话", tags=["聊天模型"])
async def chat_stream(request: ChatRequest):
    """流式聊天对话接口"""
    try:
        model = llm_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        async def generate():
            stream_generator, _ = await model.chat_stream(
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                user_question=request.user_question
            )
            async for chunk in stream_generator:
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式聊天请求失败: {str(e)}")


# ==================== 计算机视觉模型API ====================

@router.post("/cv/describe", summary="图像描述", tags=["计算机视觉模型"])
async def describe_image(request: ImageDescribeRequest):
    """图像描述接口"""
    try:
        model = cv_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 解码base64图像
        image_data = base64.b64decode(request.image_base64)
        
        result = await model.describe(image_data)
        
        return {
            "description": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像描述失败: {str(e)}")


@router.post("/cv/describe-with-prompt", summary="带提示词的图像描述", tags=["计算机视觉模型"])
async def describe_image_with_prompt(request: ImageDescribeWithPromptRequest):
    """带提示词的图像描述接口"""
    try:
        model = cv_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 解码base64图像
        image_data = base64.b64decode(request.image_base64)
        
        result = await model.describe_with_prompt(image_data, request.prompt)
        
        return {
            "description": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"带提示词图像描述失败: {str(e)}")


@router.post("/cv/chat", summary="图像聊天", tags=["计算机视觉模型"])
async def image_chat(request: ImageChatRequest):
    """图像聊天接口"""
    try:
        model = cv_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 解码base64图像
        image_data = base64.b64decode(request.image_base64)
        
        result = await model.chat(image_data, request.user_question)
        
        return {
            "response": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像聊天失败: {str(e)}")


@router.post("/cv/chat/stream", summary="图像流式聊天", tags=["计算机视觉模型"])
async def image_chat_stream(request: ImageChatRequest):
    """图像流式聊天接口"""
    try:
        model = cv_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 解码base64图像
        image_data = base64.b64decode(request.image_base64)
        
        async def generate():
            async for chunk in model.chat_stream(image_data, request.user_question):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像流式聊天失败: {str(e)}")


# ==================== 嵌入模型API ====================

@router.post("/embedding/encode", response_model=EmbeddingResponse, summary="文本编码", tags=["嵌入模型"])
async def encode_texts(request: EmbeddingRequest):
    """文本编码接口"""
    try:
        model = embedding_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        embeddings = await model.encode_texts(request.texts)
        
        return EmbeddingResponse(
            embeddings=embeddings,
            token_count=sum(len(text.split()) for text in request.texts)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本编码失败: {str(e)}")


@router.post("/embedding/encode-query", summary="查询文本编码", tags=["嵌入模型"])
async def encode_query(request: ModelRequest, query: str = Form(...)):
    """查询文本编码接口"""
    try:
        model = embedding_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        embedding = await model.encode_query(query)
        
        return {
            "embedding": embedding,
            "token_count": len(query.split()),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询文本编码失败: {str(e)}")


# ==================== 重排序模型API ====================

@router.post("/rerank/similarity", response_model=RerankResponse, summary="相似度计算", tags=["重排序模型"])
async def calculate_similarity(request: RerankRequest):
    """相似度计算接口"""
    try:
        model = rerank_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        similarities = await model.similarity(request.query, request.texts)
        
        return RerankResponse(
            similarities=similarities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似度计算失败: {str(e)}")


# ==================== 语音转文本模型API ====================

@router.post("/stt/transcribe", summary="语音转文本", tags=["语音转文本模型"])
async def transcribe_audio(audio_file: UploadFile = File(...), provider: Optional[str] = Form(None), model: Optional[str] = Form(None)):
    """语音转文本接口"""
    try:
        stt_model = stt_factory.create_model(provider, model)
        
        if not stt_model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 读取音频文件
        audio_data = await audio_file.read()
        
        # 根据模型类型处理音频数据
        if hasattr(stt_model, '_prepare_audio_input'):
            # 对于QwenSTT等需要特殊处理的模型
            audio_input = stt_model._prepare_audio_input(audio_data)
        else:
            # 对于其他模型，直接使用字节数据
            audio_input = audio_data
        
        result = await stt_model.stt(audio_input)
        
        return {
            "text": result,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音转文本失败: {str(e)}")


# ==================== 文本转语音模型API ====================

@router.post("/tts/synthesize", summary="文本转语音", tags=["文本转语音模型"])
async def synthesize_speech(request: TTSRequest):
    """文本转语音接口"""
    try:
        model = tts_factory.create_model(request.provider, request.model_name)
        
        if not model:
            raise HTTPException(status_code=400, detail="无法创建模型实例")
        
        # 调用TTS模型
        audio_gen, token_count = await model.tts(request.text, voice=request.voice)
        
        # 收集音频数据
        audio_data = b''
        for chunk in audio_gen:
            audio_data += chunk
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesized_audio.wav",
                "X-Token-Count": str(token_count),
                "X-Model-Used": f"{request.provider or 'default'}/{request.model_name or 'default'}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本转语音失败: {str(e)}")

