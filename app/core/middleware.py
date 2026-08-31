import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """为每个请求注入 request_id，记录访问日志。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # ✅ 生成 request_id
        request_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        # ✅ 绑定到 structlog 上下文（该请求内所有日志自动带上）
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start_time = time.perf_counter()

        response = await call_next(request)

        # ✅ 记录访问日志
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        logger.info(
            "http.request.completed",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        # ✅ 把 request_id 写入响应头
        response.headers["X-Request-Id"] = request_id
        return response