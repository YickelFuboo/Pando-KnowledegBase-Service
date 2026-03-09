import asyncio
import base64
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
from jose import JWTError, jwt
from app.config.settings import settings


class JWTValidationError(Exception):
    """JWT验证错误基类"""
    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class JWTValidator:
    """JWT验证器 - 从 settings 读取配置，向用户服务拉取 JWKS/配置/黑名单并验签。"""

    def __init__(
        self,
        cache_ttl: int = 3600,
        blacklist_cache_ttl: int = 300,
    ):
        """
        初始化JWT验证器

        Args:
            cache_ttl: 缓存时间（秒），默认1小时
            blacklist_cache_ttl: 黑名单缓存时间（秒），默认5分钟
        """
        self._user_service_url = (settings.auth_user_service_url or "").rstrip("/")
        self._request_timeout = settings.auth_request_timeout or 5
        self._jwks_endpoint = settings.auth_jwks_endpoint or "/.well-known/jwks.json"
        self._jwt_config_endpoint = settings.auth_jwt_config_endpoint or "/jwt-config"
        self._blacklist_endpoint = settings.auth_blacklist_endpoint or "/blacklist"
        self.cache_ttl = cache_ttl
        self.blacklist_cache_ttl = blacklist_cache_ttl
        self._jwks_cache = None
        self._jwks_cache_time = None
        self._config_cache = None
        self._config_cache_time = None
        self._blacklist_cache = None
        self._blacklist_cache_time = None
        self._client = None

    def _get_client(self) -> httpx.Client:
        """获取HTTP客户端"""
        if self._client is None:
            self._client = httpx.Client(timeout=self._request_timeout)
        return self._client

    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """检查缓存是否有效"""
        if cache_time is None:
            return False
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_ttl

    def _fetch_jwks(self) -> Dict[str, Any]:
        """获取JWKS"""
        try:
            client = self._get_client()
            url = f"{self._user_service_url}{self._jwks_endpoint}"

            response = client.get(url)
            if response.status_code == 200:
                return response.json()
            raise JWTValidationError(
                f"获取JWKS失败: {response.status_code}",
                "JWKS_FETCH_ERROR",
                {"status_code": response.status_code, "url": url}
            )
        except JWTValidationError:
            raise
        except Exception as e:
            logging.error(f"获取JWKS异常: {e}")
            raise JWTValidationError(f"获取JWKS异常: {e}", "JWKS_FETCH_EXCEPTION")

    def _fetch_jwt_config(self) -> Dict[str, Any]:
        """获取JWT配置"""
        try:
            client = self._get_client()
            url = f"{self._user_service_url}{self._jwt_config_endpoint}"

            response = client.get(url)
            if response.status_code == 200:
                result = response.json()
                return result.get("data", result)
            raise JWTValidationError(
                f"获取JWT配置失败: {response.status_code}",
                "CONFIG_FETCH_ERROR",
                {"status_code": response.status_code, "url": url}
            )
        except JWTValidationError:
            raise
        except Exception as e:
            logging.error(f"获取JWT配置异常: {e}")
            raise JWTValidationError(f"获取JWT配置异常: {e}", "CONFIG_FETCH_EXCEPTION")

    def _fetch_blacklist(self) -> List[str]:
        """获取黑名单令牌列表"""
        try:
            client = self._get_client()
            url = f"{self._user_service_url}{self._blacklist_endpoint}"

            response = client.get(url)
            if response.status_code == 200:
                result = response.json()
                return result.get("data", {}).get("blacklisted_tokens", [])
            logging.warning(f"获取黑名单失败: {response.status_code}")
            return []
        except Exception as e:
            logging.error(f"获取黑名单异常: {e}")
            return []

    def _get_blacklist_cache(self) -> List[str]:
        """获取黑名单缓存"""
        if not self._is_cache_valid(self._blacklist_cache_time):
            self._blacklist_cache = self._fetch_blacklist()
            self._blacklist_cache_time = datetime.utcnow()
        return self._blacklist_cache or []

    def get_jwks(self) -> Dict[str, Any]:
        """获取JWKS（带缓存）"""
        if not self._is_cache_valid(self._jwks_cache_time):
            self._jwks_cache = self._fetch_jwks()
            self._jwks_cache_time = datetime.utcnow()
        return self._jwks_cache

    def get_jwt_config(self) -> Dict[str, Any]:
        """获取JWT配置（带缓存）"""
        if not self._is_cache_valid(self._config_cache_time):
            self._config_cache = self._fetch_jwt_config()
            self._config_cache_time = datetime.utcnow()
        return self._config_cache

    def _is_token_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        try:
            blacklist = self._get_blacklist_cache()
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            return token_hash in blacklist
        except Exception as e:
            logging.error(f"检查黑名单异常: {e}")
            return False

    def _validate_payload_fields(self, payload: Dict[str, Any]) -> None:
        """验证JWT payload中的必需字段"""
        # 验证必需字段
        required_fields = ["sub", "username", "roles"]
        for field in required_fields:
            if field not in payload:
                raise JWTValidationError(
                    f"令牌缺少必需字段: {field}",
                    "MISSING_REQUIRED_FIELD",
                    {"missing_field": field}
                )
        
        # 验证字段类型
        if not isinstance(payload.get("roles"), list):
            raise JWTValidationError(
                "roles字段必须是数组类型",
                "INVALID_FIELD_TYPE",
                {"field": "roles", "expected_type": "list"}
            )
        
        # 验证用户状态
        if not payload.get("is_active", True):
            raise JWTValidationError(
                "用户账户已被禁用",
                "USER_INACTIVE",
                {"user_id": payload.get("sub")}
            )

    def _build_success_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """构建验证成功返回的 data"""
        return {
            "valid": True,
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "roles": payload.get("roles", []),
            "email": payload.get("email"),
            "phone": payload.get("phone"),
            "full_name": payload.get("full_name"),
            "is_superuser": payload.get("is_superuser", False),
            "is_active": payload.get("is_active", True),
            "exp": payload.get("exp"),
            "iat": payload.get("iat"),
            "iss": payload.get("iss"),
            "aud": payload.get("aud")
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """同步验证JWT令牌"""
        try:
            config = self.get_jwt_config()
            algorithm = config.get("algorithm", "HS256")
            issuer = config.get("issuer")
            audience = config.get("audience")
            
            # 获取JWKS
            jwks = self.get_jwks()
            
            # 从JWKS中获取密钥
            secret_key = None
            for key in jwks.get("keys", []):
                if key.get("kty") == "oct":
                    k = key.get("k", "")
                    k += "=" * (4 - len(k) % 4)
                    secret_key = base64.urlsafe_b64decode(k).decode("utf-8")
                    break
            
            if not secret_key:
                return {
                    "success": False,
                    "message": "无法获取验证密钥",
                    "data": {"valid": False, "reason": "no_key", "error_code": "NO_SECRET_KEY"}
                }
            
            # 验证JWT令牌
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=[algorithm],
                issuer=issuer,
                audience=audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # 验证payload字段
            self._validate_payload_fields(payload)
            
            # 检查令牌类型（可选验证）
            token_type = payload.get("type")
            if token_type and token_type != "access":
                return {
                    "success": False,
                    "message": "令牌类型错误",
                    "data": {
                        "valid": False,
                        "reason": "wrong_token_type",
                        "error_code": "INVALID_TOKEN_TYPE",
                        "expected_type": "access",
                        "actual_type": token_type
                    }
                }
            
            # 检查令牌是否在黑名单中
            if self._is_token_blacklisted(token):
                return {
                    "success": False,
                    "message": "令牌已被注销",
                    "data": {"valid": False, "reason": "token_blacklisted", "error_code": "TOKEN_BLACKLISTED"}
                }
            return {
                "success": True,
                "message": "令牌验证成功",
                "data": self._build_success_data(payload)
            }
        except JWTValidationError as e:
            logging.warning(f"JWT验证失败: {e.message} (错误码: {e.error_code})")
            return {
                "success": False,
                "message": e.message,
                "data": {"valid": False, "reason": "validation_error", "error_code": e.error_code, "details": e.details}
            }
        except JWTError as e:
            logging.warning(f"JWT解析失败: {e}")
            return {
                "success": False,
                "message": f"JWT验证失败: {str(e)}",
                "data": {"valid": False, "reason": "jwt_error", "error_code": "JWT_DECODE_ERROR"}
            }
        except Exception as e:
            logging.error(f"令牌验证异常: {e}")
            return {
                "success": False,
                "message": f"令牌验证异常: {str(e)}",
                "data": {"valid": False, "reason": "exception", "error_code": "VALIDATION_EXCEPTION"}
            }

    async def verify_token_async(self, token: str) -> Dict[str, Any]:
        """异步验证JWT令牌"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.verify_token(token))

    def extract_user_info(self, token: str) -> Dict[str, Any]:
        """从令牌中提取用户信息（不验证签名，仅解析）"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return {
                "success": True,
                "message": "提取用户信息成功",
                "data": {
                    "user_id": payload.get("sub"),
                    "username": payload.get("username"),
                    "roles": payload.get("roles", []),
                    "email": payload.get("email"),
                    "phone": payload.get("phone"),
                    "full_name": payload.get("full_name"),
                    "language": payload.get("language"),
                    "is_active": payload.get("is_active", True),
                    "is_superuser": payload.get("is_superuser", False),
                    "exp": payload.get("exp"),
                    "iat": payload.get("iat")
                }
            }
            
        except JWTError as e:
            logging.warning(f"JWT解析失败: {e}")
            return {"success": False, "message": f"JWT解析失败: {str(e)}", "data": None}

    def is_token_expired(self, token: str) -> bool:
        """检查令牌是否过期"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            exp = payload.get("exp")
            if exp is None:
                return True
            return datetime.utcnow().timestamp() > exp
        except JWTError:
            return True

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """获取令牌过期时间"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            exp = payload.get("exp")
            if exp is None:
                return None
            
            return datetime.fromtimestamp(exp)
        except JWTError:
            return None

    def refresh_cache(self) -> None:
        """刷新缓存"""
        self._jwks_cache = None
        self._jwks_cache_time = None
        self._config_cache = None
        self._config_cache_time = None
        self._blacklist_cache = None
        self._blacklist_cache_time = None

    def refresh_blacklist_cache(self) -> None:
        """刷新黑名单缓存"""
        self._blacklist_cache = None
        self._blacklist_cache_time = None

    def close(self) -> None:
        """关闭客户端连接"""
        if self._client:
            self._client.close()
            self._client = None


def create_jwt_validator(
    cache_ttl: int = 3600,
    blacklist_cache_ttl: int = 300
) -> JWTValidator:
    """创建JWT验证器"""
    return JWTValidator(cache_ttl=cache_ttl, blacklist_cache_ttl=blacklist_cache_ttl)


def verify_token_simple(token: str,) -> bool:
    """简单令牌验证（只返回是否有效）"""
    validator = JWTValidator()
    try:
        result = validator.verify_token(token)
        return result.get("success", False) and result.get("data", {}).get("valid", False)
    finally:
        validator.close()
