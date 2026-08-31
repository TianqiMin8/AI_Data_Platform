from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.config.settings import get_settings

settings = get_settings()

# 用来连接数据库
engine = create_async_engine(
    settings.database_url,
    echo=settings.is_dev,  # dev 环境打印 SQL 方便调试
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # ✅ 每次取连接前检测是否存活
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 保证多用户同时使用一session
async def get_db() -> AsyncSession:
    """FastAPI 依赖注入用的 DB session 生成器。"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise