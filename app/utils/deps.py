import logging
from typing import Any, Dict, List
from fastapi import Depends, HTTPException, status, Request
from app.utils.auth.jwt_middleware import create_jwt_dependency


# 支持的语言列表
SUPPORTED_LANGUAGES = [
    {"code": "zh-CN", "name": "简体中文", "is_default": False},
    {"code": "en-US", "name": "English", "is_default": True}
]

def get_supported_languages() -> List[dict]:
    """获取支持的语言列表"""
    return SUPPORTED_LANGUAGES

def is_supported_language(language: str) -> bool:
    """检查是否为支持的语言"""
    return language in [lang["code"] for lang in SUPPORTED_LANGUAGES]

def get_default_language() -> str:
    """获取默认语言"""
    for lang in SUPPORTED_LANGUAGES:
        if lang["is_default"]:
            return lang["code"]
    return "en-US"  # 默认返回英文 

def get_request_language(request: Request) -> str:
    """
    从请求头获取语言设置
    
    优先级：
    1. X-Language 自定义请求头
    2. Accept-Language 请求头
    3. 默认语言
    """
    # 检查自定义语言头
    custom_language = request.headers.get("X-Language")
    if custom_language and is_supported_language(custom_language):
        return custom_language
    
    # 检查标准 Accept-Language 头
    accept_language = request.headers.get("Accept-Language")
    if accept_language:
        # 解析 Accept-Language 头，格式如: "zh-CN,zh;q=0.9,en;q=0.8"
        languages = accept_language.split(",")
        for lang in languages:
            # 提取语言代码，去除质量值
            lang_code = lang.split(";")[0].strip()
            # 标准化语言代码
            if lang_code.startswith("zh"):
                lang_code = "zh-CN"
            elif lang_code.startswith("en"):
                lang_code = "en-US"
            
            if is_supported_language(lang_code):
                return lang_code
    
    # 返回默认语言
    return get_default_language()

_jwt_dependency = None

def init_auth(
    cache_ttl: int = 3600,
    blacklist_cache_ttl: int = 300
) -> None:
    """初始化认证依赖，业务服务启动时调用一次。"""
    global _jwt_dependency
    _jwt_dependency = create_jwt_dependency(cache_ttl=cache_ttl, blacklist_cache_ttl=blacklist_cache_ttl)


def _get_jwt_dependency():
    if _jwt_dependency is None:
        raise RuntimeError("请先调用 init_auth(settings) 初始化认证依赖")
    return _jwt_dependency


def get_current_user(
    user_data: Dict[str, Any] = Depends(_get_jwt_dependency)
) -> Dict[str, Any]:
    """获取当前用户信息（需已调用 init_auth）"""
    return user_data


def get_current_active_user(
    user_data: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取当前活跃用户（已启用账户）"""
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "用户账户已被禁用"}
        )
    return user_data


def get_current_superuser(
    user_data: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """获取当前超级管理员用户"""
    if not user_data.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "权限不足"}
        )
    return user_data