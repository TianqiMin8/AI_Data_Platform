import time
import psycopg2
from psycopg2 import errors as pg_errors
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

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
        classified = _classify_error(e, sql)

        logger.error(
            "sql.execution_failed",
            datasource_id=datasource.id,
            sql=sql[:200],
            error_type=classified["type"],
            error=str(e),
        )
        raise SQLExecutionError(
            classified["message"],
            db_error=str(e),
            error_type=classified["type"],
        )


# PostgreSQL 异常类型 → (错误类型, 用户可读提示)
# 顺序有意义：isinstance 按顺序匹配，子类要排在父类前面
# （QueryCanceled 是 OperationalError 的子类，必须先于连接错误判断）
PG_ERROR_TYPES: list[tuple[type[Exception], str, str]] = [
    (pg_errors.SyntaxError, "syntax_error",
     "生成的 SQL 语法错误，请尝试用不同的方式描述您的问题"),
    (pg_errors.UndefinedTable, "undefined_table",
     "查询引用了不存在的表，请确认数据源的表结构是否变更"),
    (pg_errors.UndefinedColumn, "undefined_column",
     "查询引用了不存在的字段，请确认数据源的表结构是否变更"),
    (pg_errors.UndefinedFunction, "undefined_function",
     "查询使用了不支持的函数或参数类型（例如对文本字段求和），请换一种方式描述您的问题"),
    (pg_errors.AmbiguousColumn, "ambiguous_column",
     "查询中的字段名有歧义（多张表有同名字段），请说明要查哪张表的字段"),
    (pg_errors.GroupingError, "grouping_error",
     "查询的分组方式有误，请尝试更明确地说明按什么维度统计"),
    (pg_errors.DivisionByZero, "division_by_zero",
     "计算中出现了除以零，请检查问题中的计算逻辑"),
    (pg_errors.InvalidTextRepresentation, "type_mismatch",
     "查询中的值与字段类型不匹配，请检查问题中的条件（如日期、数字格式）"),
    (pg_errors.DatatypeMismatch, "type_mismatch",
     "查询中的值与字段类型不匹配，请检查问题中的条件（如日期、数字格式）"),
    (pg_errors.InsufficientPrivilege, "permission_denied",
     "数据库用户权限不足，请检查数据源的数据库用户是否有 SELECT 权限"),
    (pg_errors.QueryCanceled, "timeout",
     "查询执行超时，请尝试缩小查询范围（如增加时间过滤条件）"),
    (psycopg2.OperationalError, "connection_error",
     "无法连接到数据库，请检查数据源的连接信息是否正确"),
]

# 错误类型 → 能否通过"把报错反馈给 LLM 重写 SQL"来修复
ERROR_RECOVERABILITY: dict[str, bool] = {
    # ── 可修复：SQL 本身写错了，LLM 看到报错能改对 ──
    "syntax_error": True,        # 语法错误、方言不对（如 PostgreSQL 里用 QUARTER()）
    "undefined_table": True,     # L2 已拦住不在 Schema 里的表，走到执行阶段多半是大小写/引号/schema 前缀问题
    "undefined_column": True,
    "undefined_function": True,  # 函数不存在或参数类型不对（SUM 文本列）
    "ambiguous_column": True,    # 加表前缀即可
    "grouping_error": True,      # 补 GROUP BY
    "division_by_zero": True,    # 用 NULLIF 包一层
    "type_mismatch": True,
    "table_not_found": True,     # 非 PostgreSQL 时字符串匹配兜底的类型
    "validation_failed": True,   # 重试中生成的 SQL 没过安全校验
    "empty_sql": True,           # 重试中 LLM 没返回 SQL
    # ── 不可修复：问题不在 SQL 上，重写多少次都一样 ──
    "permission_denied": False,  # 数据库账号没权限
    "connection_error": False,   # 数据库连不上
    "timeout": False,            # 数据量/索引问题，重写往往还是慢，而且每次要再等一个超时
    "unknown": False,            # 不认识的错误，不赌，省下 LLM 调用
}


def is_retryable(error_type: str) -> bool:
    """未登记的错误类型一律按不可修复处理。"""
    return ERROR_RECOVERABILITY.get(error_type, False)


def _classify_error(exc: Exception, sql: str) -> dict:
    """将数据库异常分类为用户可读的错误信息。

    优先按 psycopg2 的具体异常类型判断；SQLAlchemy 会把驱动异常包一层，原始异常在 .orig 上。
    拿不到 psycopg2 异常时（例如 MySQL 数据源），退回字符串匹配。
    """
    db_exc = exc.orig if isinstance(exc, DBAPIError) else exc

    if isinstance(db_exc, psycopg2.Error):
        for error_class, error_type, message in PG_ERROR_TYPES:
            if isinstance(db_exc, error_class):
                return {"type": error_type, "message": message}
        return {"type": "unknown", "message": "查询执行失败，请稍后重试"}

    return _classify_error_by_message(str(exc).lower(), sql)


def _classify_error_by_message(error_msg: str, sql: str) -> dict:
    """（旧实现）用字符串匹配做错误分类。保留用于非 PostgreSQL 数据源兜底，以及和异常类型分类做准确率对比。"""

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