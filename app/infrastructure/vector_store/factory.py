from typing import Optional
import logging
from app.config import settings
from app.infrastructure.vector_store.base import VectorStoreConnection
from app.infrastructure.vector_store.es_conn import ESConnection
from app.infrastructure.vector_store.opensearch_conn import OSConnection


class VectorStoreFactory:
    """向量存储工厂类"""
    
    def __init__(self):
        self._connection: Optional[VectorStoreConnection] = None
        self._connection_type: Optional[str] = None
        self._mapping_name: Optional[str] = None
    
    def create_connection(self, db_type: str = None, mapping_name: str = None) -> VectorStoreConnection:
        """
        创建向量存储连接：
        1. 维护连接状态和类型信息
        2. 支持连接复用和状态查询
        3. 自动进行异步初始化
        
        Args:
            db_type: 数据库类型，如果为None则从配置读取
            mapping_name: 映射文件名称，如果为None则从配置读取
        
        Returns:
            VectorStoreConnection: 已初始化的向量存储连接实例
        """
        # 使用配置中的默认值
        actual_db_type = db_type or settings.vector_store_engine
        actual_mapping_name = mapping_name or settings.vector_store_mapping
        
        db_type = actual_db_type.lower()
        
        try:
            if db_type == "elasticsearch":
                connection = ESConnection(
                    hosts=settings.es_hosts,
                    username=settings.es_username,
                    password=settings.es_password,
                    mapping_name=actual_mapping_name,
                    verify_certs=settings.es_verify_certs,
                )
            elif db_type == "opensearch":
                connection = OSConnection(
                    hosts=settings.os_hosts,
                    username=settings.os_username,
                    password=settings.os_password,
                    mapping_name=actual_mapping_name
                )
            else:
                raise ValueError(f"不支持的数据库类型: {db_type}")
            
            # 保存连接信息
            self._connection = connection
            self._connection_type = actual_db_type
            self._mapping_name = actual_mapping_name
            
            logging.info(f"向量存储连接创建并初始化成功: {actual_db_type}")
            return connection
            
        except Exception as e:
            logging.error(f"创建向量存储连接失败: {e}")
            raise

# 全局工厂实例
_vector_store_factory = VectorStoreFactory()

# 全局连接变量 - 异步创建并初始化
VECTOR_STORE_CONN = _vector_store_factory.create_connection()
