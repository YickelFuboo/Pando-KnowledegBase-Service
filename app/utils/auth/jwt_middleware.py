import logging
from typing import Any, Callable, List, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from .jwt_validator import JWTValidator


class JWTAuthMiddleware:
    """JWT认证中间件 - 供业务微服务使用，从 settings 读取配置。"""

    def __init__(
        self,
        exclude_paths: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None,
        cache_ttl: int = 3600,
        blacklist_cache_ttl: int = 300,
        on_auth_failed: Optional[Callable] = None
    ):
        """
        初始化JWT认证中间件

        Args:
            exclude_paths: 排除的路径列表，如 ["/health", "/docs"]
            exclude_methods: 排除的HTTP方法列表，如 ["GET", "OPTIONS"]
            cache_ttl: 缓存时间（秒）
            blacklist_cache_ttl: 黑名单缓存时间（秒）
            on_auth_failed: 认证失败时的回调函数
        """
        self.validator = JWTValidator(cache_ttl=cache_ttl, blacklist_cache_ttl=blacklist_cache_ttl)
        self.exclude_paths = exclude_paths or []
        self.exclude_methods = exclude_methods or ["OPTIONS"]
        self.on_auth_failed = on_auth_failed
    
    async def __call__(self, request: Request, call_next):
        """中间件处理函数"""
        # 检查是否排除当前路径
        if self._should_exclude(request):
            return await call_next(request)
        
        # 获取Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return self._handle_auth_failed("缺少Authorization头", status.HTTP_401_UNAUTHORIZED)
        
        # 检查Bearer格式
        if not auth_header.startswith("Bearer "):
            return self._handle_auth_failed("Authorization格式错误", status.HTTP_401_UNAUTHORIZED)

        token = auth_header[7:]
        result = await self.validator.verify_token_async(token)
        if not result.get("success") or not result.get("data", {}).get("valid"):
            return self._handle_auth_failed(
                result.get("message", "令牌验证失败"),
                status.HTTP_401_UNAUTHORIZED
            )
        
        # 将用户信息添加到请求状态中
        user_data = result.get("data", {})
        request.state.user_id = user_data.get("user_id")
        request.state.username = user_data.get("username")
        request.state.roles = user_data.get("roles", [])
        request.state.is_superuser = user_data.get("is_superuser", False)
        request.state.token_data = user_data
        
        # 继续处理请求
        return await call_next(request)
    
    def _should_exclude(self, request: Request) -> bool:
        """检查是否应该排除当前请求"""
        # 检查路径
        for path in self.exclude_paths:
            if request.url.path.startswith(path):
                return True
        
        # 检查HTTP方法
        if request.method in self.exclude_methods:
            return True
        
        return False
    
    def _handle_auth_failed(self, message: str, status_code: int):
        """处理认证失败"""
        if self.on_auth_failed:
            return self.on_auth_failed(message, status_code)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "message": message,
                "error_code": "AUTH_FAILED"
            }
        )

class JWTAuthDependency:
    """JWT认证依赖 - 供FastAPI路由使用，从 settings 读取配置。"""

    def __init__(
        self,
        cache_ttl: int = 3600,
        blacklist_cache_ttl: int = 300
    ):
        """
        初始化JWT认证依赖

        Args:
            cache_ttl: 缓存时间（秒）
            blacklist_cache_ttl: 黑名单缓存时间（秒）
        """
        self.validator = JWTValidator(cache_ttl=cache_ttl, blacklist_cache_ttl=blacklist_cache_ttl)

    async def __call__(self, request: Request):
        """依赖函数"""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="缺少Authorization头",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization格式错误",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token = auth_header[7:]
        result = await self.validator.verify_token_async(token)
        if not result.get("success") or not result.get("data", {}).get("valid"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.get("message", "令牌验证失败"),
                headers={"WWW-Authenticate": "Bearer"}
            )
        return result.get("data", {})
    
    def close(self):
        """关闭验证器"""
        self.validator.close()

def create_jwt_middleware(
    exclude_paths: Optional[List[str]] = None,
    exclude_methods: Optional[List[str]] = None,
    cache_ttl: int = 3600,
    blacklist_cache_ttl: int = 300
) -> JWTAuthMiddleware:
    """创建JWT认证中间件，需传入 settings。"""
    return JWTAuthMiddleware(
        exclude_paths=exclude_paths,
        exclude_methods=exclude_methods,
        cache_ttl=cache_ttl,
        blacklist_cache_ttl=blacklist_cache_ttl
    )

def create_jwt_dependency(
    cache_ttl: int = 3600,
    blacklist_cache_ttl: int = 300
) -> JWTAuthDependency:
    """创建JWT认证依赖，需传入 settings。业务服务在启动时调用并注入到 deps。"""
    return JWTAuthDependency(cache_ttl=cache_ttl, blacklist_cache_ttl=blacklist_cache_ttl)