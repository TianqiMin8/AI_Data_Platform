from fastapi import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

class AppError(Exception):
    """所有业务异常的基类。"""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str, id: str | int = ""):
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource} 不存在" + (f" (id={id})" if id else ""),
            status_code=404,
        )


class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(code="VALIDATION_ERROR", message=message, status_code=422)


class LLMError(AppError):
    def __init__(self, message: str = "LLM 服务暂时不可用"):
        super().__init__(code="LLM_ERROR", message=message, status_code=502)


class SQLGenerationError(AppError):
    def __init__(self, message: str = "SQL 生成失败"):
        super().__init__(code="SQL_GENERATION_ERROR", message=message, status_code=500)


class PermissionDeniedError(AppError):
    def __init__(self, message: str = "权限不足"):
        super().__init__(code="PERMISSION_DENIED", message=message, status_code=403)

class LLMError(AppError):
        def __init__(self, message: str = "LLM 服务暂时不可用"):
            super().__init__(code="LLM_ERROR", message=message, status_code=502)


class SQLGenerationError(AppError):
    def __init__(self, message: str = "SQL 生成失败"):
        super().__init__(code="SQL_GENERATION_ERROR", message=message, status_code=500)


class SQLExecutionError(AppError):
    def __init__(self, message: str = "SQL 执行失败"):
        super().__init__(code="SQL_EXECUTION_ERROR", message=message, status_code=500)


# app/core/errors.py 追加

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.core.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器。"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        request_id = getattr(request.state, "request_id", "")
        logger.warning(
            "app.error",
            error_code=exc.code,
            error_message=exc.message,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", "")

        logger.warning(
            "http.error",
            status_code=exc.status_code,
            error_message=str(exc.detail),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "request_id": request_id,
                }
            },
        )


    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "")
        logger.error(
            "app.unhandled_error",
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务内部错误，请稍后重试",
                    "request_id": request_id,
                }
            },
        )


    
            