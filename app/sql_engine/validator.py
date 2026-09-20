from dataclasses import dataclass
import sqlglot
from sqlglot import exp

from app.core.logging import get_logger

logger = get_logger(__name__)

# 禁止的 SQL 语句类型
BLOCKED_STATEMENT_TYPES = (
    exp.Delete,
    exp.Drop,
    exp.Insert,
    exp.Update,
    exp.Create,
    exp.Alter,
    exp.Grant,
    exp.Command,       # TRUNCATE / VACUUM 等
)

# 禁止的函数名（小写）
DANGEROUS_FUNCTIONS = {
    "pg_sleep",
    "sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_read_file",
    "pg_write_file",
    "lo_import",
    "lo_export",
    "dblink",
    "dblink_exec",
    "copy",
}


@dataclass
class ValidationResult:
    is_safe: bool
    level: str = ""       # L1 / L2 / L3 / L4
    reason: str = ""
    blocked_detail: str = ""


def validate_sql(
    sql: str,
    allowed_tables: list[str] | None = None,
    db_type: str = "postgresql",
    max_subquery_depth: int = 4,
    max_join_count: int = 6,
) -> ValidationResult:
    """对 LLM 生成的 SQL 进行四层安全校验。"""

    # ── L1：语句类型检查 ──
    try:
        parsed = sqlglot.parse(sql, read=db_type)
    except Exception as e:
        return ValidationResult(
            is_safe=False,
            level="L1",
            reason="SQL 语法解析失败",
            blocked_detail=str(e),
        )

    if not parsed:
        return ValidationResult(
            is_safe=False, level="L1", reason="空 SQL", blocked_detail="解析结果为空"
        )

    # 检查是否有多条语句（防止 SQL 注入拼接）
    if len(parsed) > 1:
        return ValidationResult(
            is_safe=False,
            level="L1",
            reason="检测到多条 SQL 语句",
            blocked_detail="只允许单条 SELECT 语句",
        )

    statement = parsed[0]

    # 检查根节点是否为 SELECT
    if isinstance(statement, BLOCKED_STATEMENT_TYPES):
        stmt_type = type(statement).__name__
        return ValidationResult(
            is_safe=False,
            level="L1",
            reason=f"不允许的 SQL 操作：{stmt_type}",
            blocked_detail=f"系统只允许 SELECT 查询，检测到{stmt_type}",
        )

    if not isinstance(statement, exp.Select):
        stmt_type = type(statement).__name__
        return ValidationResult(
            is_safe=False,
            level="L1",
            reason=f"非 SELECT 语句：{stmt_type}",
            blocked_detail=f"系统只允许 SELECT 查询",
        )

    # ── L2：表名授权检查 ──
    if allowed_tables is not None:
        allowed_set = {t.lower() for t in allowed_tables}
        referenced_tables = set()
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower()
            referenced_tables.add(table_name)

        unauthorized = referenced_tables - allowed_set
        if unauthorized:
            return ValidationResult(
                is_safe=False,
                level="L2",
                reason="查询了未授权的表",
                blocked_detail=f"不允许查询的表：{', '.join(sorted(unauthorized))}",
            )

    # ── L3：危险函数检查 ──
    for func_node in statement.find_all(exp.Anonymous, exp.Func):
        func_name = ""
        if isinstance(func_node, exp.Anonymous):
            func_name = func_node.name.lower()
        elif hasattr(func_node, "sql_name"):
            func_name = func_node.sql_name().lower()

        if func_name in DANGEROUS_FUNCTIONS:
            return ValidationResult(
                is_safe=False,
                level="L3",
                reason=f"检测到危险函数：{func_name}",
                blocked_detail=f"函数{func_name} 被禁止使用",
            )

    # ── L4：复杂度限制 ──
    # 子查询深度
    depth = _measure_subquery_depth(statement)
    if depth > max_subquery_depth:
        return ValidationResult(
            is_safe=False,
            level="L4",
            reason=f"子查询嵌套过深（{depth} 层，最大{max_subquery_depth}）",
            blocked_detail="请简化查询，减少子查询嵌套层数",
        )

    # JOIN 数量
    join_count = len(list(statement.find_all(exp.Join)))
    if join_count > max_join_count:
        return ValidationResult(
            is_safe=False,
            level="L4",
            reason=f"JOIN 数量过多（{join_count} 个，最大{max_join_count}）",
            blocked_detail="请简化查询，减少 JOIN 数量",
        )

    logger.info("sql.validation.passed", sql_length=len(sql))
    return ValidationResult(is_safe=True)


def _measure_subquery_depth(node, current_depth=0) -> int:
    """递归测量子查询嵌套深度。"""
    max_depth = current_depth
    for child in node.iter_expressions():
        if isinstance(child, exp.Subquery) or isinstance(child, exp.Select):
            child_depth = _measure_subquery_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        else:
            child_depth = _measure_subquery_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
    return max_depth