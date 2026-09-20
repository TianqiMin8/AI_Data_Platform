import time
from sqlalchemy import create_engine, text

from app.models.datasource import DataSource
from app.services.datasource import _build_sync_url
from app.core.security import decrypt_value
from app.core.errors import SQLExecutionError
from app.core.logging import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


class SQLExecutionResult:
    """SQL 执行结果的标准化封装。"""

    def __init__(
        self,
        columns: list[str],
        rows: list[dict],
        row_count: int,
        total_row_count: int,
        execution_ms: float,
        truncated: bool = False,
    ):
        self.columns = columns
        self.rows = rows
        self.row_count = row_count
        self.total_row_count = total_row_count
        self.execution_ms = execution_ms
        self.truncated = truncated

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "total_row_count": self.total_row_count,
            "execution_ms": self.execution_ms,
            "truncated": self.truncated,
        }


def execute_sql(datasource: DataSource, sql: str) -> SQLExecutionResult:
    """连接用户数据库执行 SQL，返回标准化结果。"""

    settings = get_settings()
    plain_password = decrypt_value(datasource.encrypted_password)
    url = _build_sync_url(datasource, plain_password)

    try:
        engine = create_engine(url, connect_args={"connect_timeout": 5})

        start_time = time.perf_counter()

        with engine.connect() as conn:
            if datasource.db_type == "postgresql":
                timeout_ms = settings.sql_timeout_seconds * 1000
                conn.execute(text(f"SET statement_timeout ={timeout_ms}"))

            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(settings.sql_max_rows)]
            row_count = result.rowcount

        engine.dispose()
        execution_ms = round((time.perf_counter() - start_time) * 1000, 1)

        total_row_count = row_count if row_count >= 0 else len(rows)
        truncated = total_row_count > settings.sql_max_rows

        logger.info(
            "sql.executed",
            datasource_id=datasource.id,
            row_count=len(rows),
            execution_ms=execution_ms,
        )

        return SQLExecutionResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            total_row_count=total_row_count,
            execution_ms=execution_ms,
            truncated=truncated,
        )

    except Exception as e:
        error_msg = str(e).lower()
        classified = _classify_error(error_msg, sql)

        logger.error(
            "sql.execution_failed",
            datasource_id=datasource.id,
            sql=sql[:200],
            error_type=classified["type"],
            error=str(e),
        )
        raise SQLExecutionError(classified["message"])


def _classify_error(error_msg: str, sql: str) -> dict:
    """将数据库报错分类为用户可读的错误信息。"""

    if "syntax error" in error_msg:
        return {
            "type": "syntax_error",
            "message": "生成的 SQL 语法错误，请尝试用不同的方式描述您的问题",
        }

    if "does not exist" in error_msg or "relation" in error_msg:
        return {
            "type": "table_not_found",
            "message": "查询引用了不存在的表或列，请确认数据源的表结构是否变更",
        }

    if "permission denied" in error_msg:
        return {
            "type": "permission_denied",
            "message": "数据库用户权限不足，请检查数据源的数据库用户是否有 SELECT 权限",
        }

    if "timeout" in error_msg or "cancel" in error_msg:
        return {
            "type": "timeout",
            "message": "查询执行超时，请尝试缩小查询范围（如增加时间过滤条件）",
        }

    if "connection" in error_msg or "connect" in error_msg:
        return {
            "type": "connection_error",
            "message": "无法连接到数据库，请检查数据源的连接信息是否正确",
        }

    return {
        "type": "unknown",
        "message": f"查询执行失败，请稍后重试",
    }