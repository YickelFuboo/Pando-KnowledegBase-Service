from logging.config import fileConfig
import os
import sys
from urllib.parse import quote_plus
from sqlalchemy.engine import URL

from sqlalchemy import engine_from_config, create_engine
from sqlalchemy import pool

from alembic import context

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 仅从项目根目录的 env 文件加载配置（绝对路径，与启动位置无关）
_project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_env_path = os.path.join(_project_root, "env")

_env_vars = {}
if os.path.exists(_env_path):
    for enc in ("utf-8", "gbk"):
        try:
            with open(_env_path, "r", encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key, value = key.strip(), value.strip()
                        if key:
                            _env_vars[key] = value
            break
        except (UnicodeDecodeError, Exception):
            continue
for k, v in _env_vars.items():
    if v is not None:
        os.environ[k] = v

# 布尔型环境变量：若被系统设为 "10"/"1"/"0"，规范为 pydantic 可解析的 true/false，避免 ValidationError
_bool_keys = (
    "DEBUG", "MINIO_SECURE", "S3_USE_SSL", "REDIS_SSL", "REDIS_DECODE_RESPONSES",
    "REDIS_RETRY_ON_TIMEOUT", "LANGFUSE_ENABLED", "AGENT_SESSION_USE_LOCAL_STORAGE",
)
for _k in _bool_keys:
    _val = os.environ.get(_k, "")
    if _val in ("10", "1"):
        os.environ[_k] = "true"
    elif _val == "0":
        os.environ[_k] = "false"

# 导入模型和配置（Base 统一使用 infrastructure/database；SessionRecord 单独导入避免循环依赖）
from app.infrastructure.database.models_base import Base
from app.domains.models import *
from app.agent_frame.session.models import SessionRecord
from app.config.settings import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

def _get_env(key: str, default: str = "") -> str:
    # 以 env 文件为准；需临时覆盖时可在终端设置： $env:POSTGRESQL_PASSWORD="xxx"
    return _env_vars.get(key) or os.environ.get(key, default) or default

# 设置数据库URL（来自项目根目录 env 文件）
try:
    database_type = (_get_env("DATABASE_TYPE") or "postgresql").strip().lower()
    db_name = _get_env("DB_NAME", "pando_knowledgevector_service")
    if database_type == "postgresql":
        sync_url = URL.create(
            drivername="postgresql+psycopg",
            username=_get_env("POSTGRESQL_USER", "postgres"),
            password=_get_env("POSTGRESQL_PASSWORD", ""),
            host=_get_env("POSTGRESQL_HOST", "localhost"),
            port=int(_get_env("POSTGRESQL_PORT", "5432")),
            database=db_name,
        )
    elif database_type == "mysql":
        sync_url = URL.create(
            drivername="mysql",
            username=_get_env("MYSQL_USER", "root"),
            password=_get_env("MYSQL_PASSWORD", ""),
            host=_get_env("MYSQL_HOST", "localhost"),
            port=int(_get_env("MYSQL_PORT", "3306")),
            database=db_name,
        )
    else:
        sync_url = "sqlite:///./koalawiki.db"
    config.set_main_option("sqlalchemy.url", str(sync_url))
except Exception as e:
    import logging
    logging.error(f"构建数据库URL失败: {e}")
    raise

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    database_type = (_get_env("DATABASE_TYPE") or "postgresql").strip().lower()
    if database_type == "postgresql":
        import psycopg
        host = _env_vars.get("POSTGRESQL_HOST", "localhost")
        if str(host).lower() in ("localhost", "127.0.0.1"):
            host = "127.0.0.1"
        def _pg_creator():
            return psycopg.connect(
                host=host,
                port=int(_env_vars.get("POSTGRESQL_PORT", "5432")),
                dbname=_env_vars.get("DB_NAME", "pando_knowledgevector_service"),
                user=_env_vars.get("POSTGRESQL_USER", "postgres"),
                password=_env_vars.get("POSTGRESQL_PASSWORD", ""),
            )
        connectable = create_engine(
            "postgresql+psycopg://",
            creator=_pg_creator,
            poolclass=pool.NullPool,
        )
    else:
        url_str = config.get_main_option("sqlalchemy.url")
        connectable = create_engine(url_str, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
