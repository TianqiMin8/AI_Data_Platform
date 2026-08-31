from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config.settings import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.errors import register_exception_handlers
from app.core.middleware import RequestContextMiddleware
from app.api.router import api_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 启动时
    settings = get_settings()
    logger.info(
        "app.starting",
        environment=settings.environment,
        version=settings.app_version,
    )
    yield
    # ✅ 关闭时
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    # 日志初始化（必须最先执行）
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.is_dev else None,  # 生产环境关闭 Swagger
        lifespan=lifespan,
    )

    # 中间件（注册顺序 = 执行顺序的逆序）
    app.add_middleware(RequestContextMiddleware)

    # 异常处理
    register_exception_handlers(app)

    # 路由
    app.include_router(api_router)

    return app


app = create_app()