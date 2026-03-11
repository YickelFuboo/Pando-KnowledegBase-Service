import threading
import uvicorn
import logging
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.logger import set_log_level, setup_logging
from app.config.settings import settings, APP_NAME, APP_VERSION, APP_DESCRIPTION
from app.middleware.logging import logging_middleware
from app.infrastructure.celery.app import celery_app
from app.infrastructure.database import close_db, health_check_db
from app.infrastructure.storage import STORAGE_CONN
from app.infrastructure.vector_store import VECTOR_STORE_CONN
from app.infrastructure.redis import REDIS_CONN
from app.utils.auth.jwt_middleware import create_jwt_middleware
from app.domains.api import kb, document, kb_qa, llm_chat, concept
from app.infrastructure.llms.api import llms
from app.agent_frame.session.api import router as sessions


# 创建FastAPI应用
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
    }
)

# 确保日志配置在应用启动时被正确设置
setup_logging()

#==================================
# 注册所有路由器
#==================================
app.include_router(kb.router, prefix="/api/v1", tags=["知识库管理"])
app.include_router(document.router, prefix="/api/v1", tags=["文档管理"])
app.include_router(kb_qa.router, prefix="/api/v1", tags=["问答服务"])
app.include_router(llms.router, prefix="/api/v1", tags=["模型服务"])
app.include_router(sessions, prefix="/api/v1", tags=["会话管理"])
app.include_router(llm_chat.router, prefix="/api/v1", tags=["LLM聊天"])
app.include_router(concept.router, prefix="/api/v1", tags=["概念管理"])

#==================================
# 配置中间件
#==================================
# 配置CORS中间件 - 直接使用FastAPI内置的CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志中间件 - 直接使用全局中间件实例
app.add_middleware(logging_middleware)

# 添加JWT中间件到应用
#app.middleware("http")(create_jwt_middleware)

def run_celery_worker():
    """在独立线程中运行 Celery Worker"""
    try:
        # 在debug模式下使用debug日志级别，否则使用info
        log_level = 'debug' if settings.debug else 'info'
        celery_app.worker_main(['worker', f'--loglevel={log_level}', '--concurrency=1', '-Q', 'document,default'])
    except Exception as e:
        logging.error(f"Celery Worker 启动失败: {e}")

#==================================
# 初始化基础设施
#==================================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    try:      
        logging.info("开始应用启动流程...")
        
        # 在debug模式下启动Celery Worker
        """ if settings.debug:
            logging.info("🔧 开发模式：准备启动 Celery Worker...")
            try:
                t = threading.Thread(target=run_celery_worker, daemon=True)
                t.start()
                await asyncio.sleep(0.1)
                logging.info("✅ Celery Worker 已在后台线程启动（开发模式）")
            except Exception as worker_error:
                logging.warning(f"⚠️ Celery Worker 启动失败，但不影响主应用: {worker_error}")
        else:
            logging.info("🏭 生产模式：跳过 Celery Worker 启动")
        """

        logging.info(f"{APP_NAME} v{APP_VERSION} 启动成功")

    except Exception as e:
        logging.error(f"应用启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    try:
        # 关闭数据库连接
        await close_db()
        
        # 关闭存储连接
        if STORAGE_CONN and hasattr(STORAGE_CONN, 'close'):
            try:
                await STORAGE_CONN.close()
            except Exception as e:
                logging.warning(f"关闭存储连接时出错: {e}")
        logging.info("存储连接已关闭")

        # 关闭向量存储连接
        if VECTOR_STORE_CONN and hasattr(VECTOR_STORE_CONN, 'close'):
            try:
                await VECTOR_STORE_CONN.close()
            except Exception as e:
                logging.warning(f"关闭向量存储连接时出错: {e}")
        logging.info("向量存储连接已关闭")

        # 关闭Redis连接
        if REDIS_CONN and hasattr(REDIS_CONN, 'close'):
            try:
                await REDIS_CONN.close()
            except Exception as e:
                logging.warning(f"关闭Redis连接时出错: {e}")
        logging.info("Redis连接已关闭")
        
    except Exception as e:
        logging.error(f"关闭连接失败: {e}")
    
    logging.info("应用正在关闭...")

# 根路径
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api/v1"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 基础服务状态检查
        health_status = {
            "status": "healthy",
            "service": APP_NAME,
            "version": APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "environment": "development" if settings.debug else "production"
        }
    
        # 检查数据库连接健康状态
        db_healthy = await health_check_db()
        health_status["database"] = "healthy" if db_healthy else "unhealthy"

        # 检查存储连接健康状态
        storage_healthy = False
        if STORAGE_CONN and hasattr(STORAGE_CONN, 'health_check'):
            storage_healthy = await STORAGE_CONN.health_check()
        health_status["storage"] = "healthy" if storage_healthy else "unhealthy"
        
        # 检查向量存储连接健康状态
        vector_healthy = False
        if VECTOR_STORE_CONN and hasattr(VECTOR_STORE_CONN, 'health_check'):
            vector_healthy = await VECTOR_STORE_CONN.health_check()
        health_status["vector_store"] = "healthy" if vector_healthy else "unhealthy"
        
        # 检查Redis连接健康状态
        redis_healthy = False
        if REDIS_CONN and hasattr(REDIS_CONN, 'health_check'):
            redis_healthy = await REDIS_CONN.health_check()
        health_status["redis"] = "healthy" if redis_healthy else "unhealthy"
        
        # 如果任何服务不健康，整体状态设为不健康
        if not db_healthy or not storage_healthy or not vector_healthy or not redis_healthy:
            health_status["status"] = "unhealthy"
                
        return health_status
        
    except Exception as e:
        logging.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="服务不健康")

@app.post("/log-level")
async def change_log_level(level: str = Query(..., description="日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL")):
    """动态设置日志级别"""
    try:
        set_log_level(level)
        current_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
        return {
            "message": f"日志级别已设置为 {level.upper()}",
            "current_level": current_level
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/log-level")
async def get_log_level():
    """获取当前日志级别"""
    current_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
    return {
        "current_level": current_level
    }

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误"}
    )

def main():
    """主函数，用于启动服务器"""
    uvicorn.run(
        "app.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.debug
    )

if __name__ == "__main__":
    main() 