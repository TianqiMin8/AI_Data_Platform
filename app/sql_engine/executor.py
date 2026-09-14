import time
from sqlalchemy import create_engine, text

from app.models.datasource import DataSource
from app.services.datasource import _build_sync_url
from app.core.security import decrypt_value
from app.core.errors import SQLExecutionError
from app.core.logging import get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


def execute_sql(datasource: DataSource, sql: str) -> dict:
    """连接用户数据库执行 SQL，返回格式化的结果。"""

    settings = get_settings()
    plain_password = decrypt_value(datasource.encrypted_password)
    url = _build_sync_url(datasource, plain_password)

    try:
        engine = create_engine(
            url,
            connect_args={"connect_timeout": 5},
        )

        start_time = time.perf_counter()

        with engine.connect() as conn:
            # 设置语句超时（PostgreSQL 特有）
            if datasource.db_type == "postgresql":
                timeout_ms = settings.sql_timeout_seconds * 1000
                conn.execute(text(f"SET statement_timeout ={timeout_ms}"))

            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(settings.sql_max_rows)]
            row_count = result.rowcount

        engine.dispose()
        execution_ms = round((time.perf_counter() - start_time) * 1000, 1)

        logger.info(
            "sql.executed",
            datasource_id=datasource.id,
            row_count=len(rows),
            total_rows=row_count,
            execution_ms=execution_ms,
        )

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "total_row_count": row_count if row_count >= 0 else len(rows),
            "execution_ms": execution_ms,
            "truncated": row_count > settings.sql_max_rows if row_count >= 0 else False,
        }

    except Exception as e:
        logger.error(
            "sql.execution_failed",
            datasource_id=datasource.id,
            sql=sql[:200],
            error=str(e),
        )
        raise SQLExecutionError(f"SQL 执行失败：{str(e)}")