from dataclasses import dataclass
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

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

# DataSource.db_type → sqlglot 方言名（sqlglot 只认 "postgres"，不认 "postgresql"）
SQLGLOT_DIALECTS = {
    "postgresql": "postgres",
    "mysql": "mysql",
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
    allowed_columns: dict[str, list[str]] | None = None,
    sensitive_columns: list[str] | None = None,
) -> ValidationResult:
    """对 LLM 生成的 SQL 进行四层安全校验。

    allowed_columns: 表名 → 列名列表（来自 Schema 自省），传入时启用 L2 列名校验。
    sensitive_columns: 敏感列关键字（全局 + 数据源配置），传入时启用 L2 敏感列过滤；
        SELECT * 需要 allowed_columns 才能展开检查。
    """

    # ── L1：语句类型检查 ──
    try:
        parsed = sqlglot.parse(sql, read=SQLGLOT_DIALECTS.get(db_type, db_type))
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

    # ── L2 扩展：列名校验 ──
    if allowed_columns is not None:
        unknown_columns = _find_unknown_columns(statement, allowed_columns)
        if unknown_columns:
            return ValidationResult(
                is_safe=False,
                level="L2",
                reason="引用了不存在的列",
                blocked_detail=f"Schema 中不存在的列：{', '.join(sorted(unknown_columns))}",
            )

    # ── L2 扩展：敏感列过滤 ──
    if sensitive_columns:
        hit_columns = _find_sensitive_columns(statement, sensitive_columns, allowed_columns)
        if hit_columns:
            return ValidationResult(
                is_safe=False,
                level="L2",
                reason="该列包含敏感信息，不允许查询",
                blocked_detail=f"敏感列：{', '.join(sorted(hit_columns))}",
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


def _find_unknown_columns(
    statement: exp.Expression, allowed_columns: dict[str, list[str]]
) -> set[str]:
    """找出 SQL 中引用了、但在 Schema 里不存在的列。

    规则：
    - SELECT * / COUNT(*) 不是具体列引用，不校验（展开后必然是表里已有的列）
    - SELECT 里定义的别名（AS total_sales）可以在 ORDER BY / GROUP BY 中被引用，视为合法
    - 带表前缀的列（o.amount）：通过表别名映射回真实表，再查该表的列
    - 不带前缀的列：只要在 FROM/JOIN 涉及的任意一张真实表里存在即可
    - 来自子查询/CTE 的列无法对照原始 Schema，跳过（子查询内部会作为独立 scope 单独校验）

    用 sqlglot 的 scope 分析逐层处理：每一层 SELECT 只看自己 FROM/JOIN 的来源，
    这样外层引用子查询输出列不会误报，子查询内部引用不存在的列也不会漏报。
    """
    columns_by_table = {
        table.lower(): {c.lower() for c in cols}
        for table, cols in allowed_columns.items()
    }

    unknown: set[str] = set()
    for scope in traverse_scope(statement):
        # 本层来源：别名 → 真实表名（FROM demo_orders o → {"o": "demo_orders"}）；子查询/CTE 单独记下
        real_tables: dict[str, str] = {}
        derived_sources: set[str] = set()
        for source_name, source in scope.sources.items():
            if isinstance(source, exp.Table):
                real_tables[source_name.lower()] = source.name.lower()
            else:
                derived_sources.add(source_name.lower())

        known_tables = [t for t in real_tables.values() if t in columns_by_table]

        # 本层 SELECT 列表里定义的输出别名
        output_aliases = set()
        if isinstance(scope.expression, exp.Select):
            output_aliases = {
                e.alias.lower() for e in scope.expression.expressions if isinstance(e, exp.Alias)
            }

        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue

            column_name = column.name.lower()
            qualifier = column.table.lower()

            if qualifier:
                real_table = real_tables.get(qualifier)
                if real_table not in columns_by_table:
                    continue  # 子查询/CTE 的列，或表不在 allowed_columns 里（交给表名校验）
                if column_name not in columns_by_table[real_table]:
                    unknown.add(f"{real_table}.{column_name}")
                continue

            if column_name in output_aliases:
                continue
            if any(column_name in columns_by_table[t] for t in known_tables):
                continue
            if derived_sources or not known_tables:
                continue  # 可能来自子查询/CTE 的输出列，或涉及的表都不在 allowed_columns 里
            unknown.add(column_name)

    return unknown


def _find_sensitive_columns(
    statement: exp.Expression,
    sensitive_keywords: list[str],
    allowed_columns: dict[str, list[str]] | None,
) -> set[str]:
    """找出 SQL 中涉及的敏感列。列名包含任一关键字（不区分大小写）即视为敏感。

    - 任何位置的显式引用都算：SELECT、WHERE、JOIN、ORDER BY……
      （WHERE hashed_password LIKE '$2b$%' 虽然不返回该列，也能逐步试探出内容）
    - SELECT * / t.* 用 allowed_columns 展开成实际列再检查；COUNT(*) 不返回列内容，不算
    """
    keywords = [k.lower() for k in sensitive_keywords if k]

    def is_sensitive(column_name: str) -> bool:
        name = column_name.lower()
        return any(k in name for k in keywords)

    columns_by_table = {
        table.lower(): cols for table, cols in (allowed_columns or {}).items()
    }

    hits: set[str] = set()
    for scope in traverse_scope(statement):
        real_tables = {
            name.lower(): source.name.lower()
            for name, source in scope.sources.items()
            if isinstance(source, exp.Table)
        }

        # 1. 显式引用的列
        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue
            if is_sensitive(column.name):
                hits.add(column.name.lower())

        # 2. SELECT 列表里的 * / t.*：展开成本层真实表的全部列
        if not isinstance(scope.expression, exp.Select):
            continue
        for projection in scope.expression.expressions:
            if isinstance(projection, exp.Star):
                star_tables = list(real_tables.values())
            elif isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                real_table = real_tables.get(projection.table.lower())
                star_tables = [real_table] if real_table else []
            else:
                continue
            for table in star_tables:
                for column_name in columns_by_table.get(table, []):
                    if is_sensitive(column_name):
                        hits.add(f"{table}.{column_name.lower()}")

    return hits


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