from app.sql_engine.validator import validate_sql

ALLOWED_TABLES = ["demo_orders", "customers"]
ALLOWED_COLUMNS = {
    "demo_orders": ["id", "customer_name", "amount", "category", "created_at"],
    "customers": ["id", "name", "city"],
}


def _validate(sql: str):
    return validate_sql(
        sql,
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
        db_type="postgresql",
    )


# ── 作业 1：列名校验 ──


def test_unknown_column_is_blocked():
    result = _validate("SELECT product_name, amount FROM demo_orders")
    assert not result.is_safe
    assert result.level == "L2"
    assert result.reason == "引用了不存在的列"
    assert "product_name" in result.blocked_detail


def test_unknown_column_in_where_is_blocked():
    result = _validate("SELECT amount FROM demo_orders WHERE status = 'paid'")
    assert not result.is_safe
    assert "status" in result.blocked_detail


def test_unknown_qualified_column_is_blocked():
    result = _validate("SELECT o.amount, o.discount FROM demo_orders o")
    assert not result.is_safe
    assert "demo_orders.discount" in result.blocked_detail


def test_column_from_wrong_table_is_blocked():
    # city 存在于 customers，但不在 demo_orders 里
    result = _validate("SELECT o.city FROM demo_orders o")
    assert not result.is_safe
    assert "demo_orders.city" in result.blocked_detail


def test_valid_columns_pass():
    result = _validate(
        "SELECT category, SUM(amount) AS total_sales FROM demo_orders "
        "GROUP BY category ORDER BY total_sales DESC"
    )
    assert result.is_safe


def test_select_star_passes():
    assert _validate("SELECT * FROM demo_orders").is_safe
    assert _validate("SELECT COUNT(*) FROM demo_orders").is_safe


def test_output_alias_can_be_referenced():
    result = _validate(
        "SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS total "
        "FROM demo_orders GROUP BY month ORDER BY month"
    )
    assert result.is_safe


def test_join_with_table_aliases_passes():
    result = _validate(
        "SELECT c.city, SUM(o.amount) FROM demo_orders o "
        "JOIN customers c ON c.name = o.customer_name GROUP BY c.city"
    )
    assert result.is_safe


def test_subquery_output_columns_pass():
    result = _validate(
        "SELECT t.customer_name, t.total FROM "
        "(SELECT customer_name, SUM(amount) AS total FROM demo_orders GROUP BY customer_name) t "
        "WHERE t.total > 2000"
    )
    assert result.is_safe


def test_unknown_column_inside_subquery_is_blocked():
    result = _validate(
        "SELECT t.total FROM (SELECT SUM(price) AS total FROM demo_orders) t"
    )
    assert not result.is_safe
    assert "price" in result.blocked_detail


def test_column_check_skipped_without_allowed_columns():
    result = validate_sql(
        "SELECT product_name FROM demo_orders",
        allowed_tables=ALLOWED_TABLES,
        db_type="postgresql",
    )
    assert result.is_safe


# ── 原有的 L1–L3 校验 ──


def test_delete_is_blocked():
    result = _validate("DELETE FROM demo_orders")
    assert not result.is_safe
    assert result.level == "L1"


def test_unauthorized_table_is_blocked():
    result = _validate("SELECT * FROM users")
    assert not result.is_safe
    assert result.level == "L2"
    assert result.reason == "查询了未授权的表"


def test_dangerous_function_is_blocked():
    result = _validate("SELECT pg_sleep(10), category FROM demo_orders")
    assert not result.is_safe
    assert result.level == "L3"


# ── 作业 4：敏感列过滤 ──

SENSITIVE_TABLES = ["users", "employees"]
SENSITIVE_ALLOWED_COLUMNS = {
    "users": ["id", "email", "hashed_password", "nickname"],
    "employees": ["id", "name", "salary", "pwd"],
}
GLOBAL_SENSITIVE = ["password", "ssn", "salary", "credit_card"]


def _validate_sensitive(sql: str, sensitive=GLOBAL_SENSITIVE):
    return validate_sql(
        sql,
        allowed_tables=SENSITIVE_TABLES,
        allowed_columns=SENSITIVE_ALLOWED_COLUMNS,
        sensitive_columns=sensitive,
        db_type="postgresql",
    )


def test_sensitive_column_in_select_is_blocked():
    # 7.4 里实际泄露密码哈希的那条 SQL：列名 hashed_password 包含关键字 password
    result = _validate_sensitive("SELECT id, email, hashed_password FROM users")
    assert not result.is_safe
    assert result.level == "L2"
    assert result.reason == "该列包含敏感信息，不允许查询"
    assert "hashed_password" in result.blocked_detail


def test_sensitive_column_in_where_is_blocked():
    result = _validate_sensitive("SELECT email FROM users WHERE hashed_password LIKE '$2b$%'")
    assert not result.is_safe


def test_sensitive_column_with_alias_is_blocked():
    result = _validate_sensitive("SELECT AVG(e.salary) AS avg_pay FROM employees e")
    assert not result.is_safe
    assert "salary" in result.blocked_detail


def test_select_star_on_table_with_sensitive_column_is_blocked():
    result = _validate_sensitive("SELECT * FROM users")
    assert not result.is_safe
    assert "users.hashed_password" in result.blocked_detail


def test_qualified_star_is_expanded():
    result = _validate_sensitive("SELECT u.* FROM users u")
    assert not result.is_safe


def test_star_inside_subquery_is_blocked():
    result = _validate_sensitive("SELECT t.email FROM (SELECT * FROM users) t")
    assert not result.is_safe


def test_count_star_is_allowed():
    assert _validate_sensitive("SELECT COUNT(*) FROM users").is_safe


def test_non_sensitive_columns_pass():
    assert _validate_sensitive("SELECT id, email, nickname FROM users").is_safe


def test_keyword_matching_misses_abbreviation():
    # 思考题：列名是 pwd 时，关键字 password 匹配不到
    assert _validate_sensitive("SELECT name, pwd FROM employees").is_safe


def test_per_datasource_keywords_extend_global():
    # 数据源额外配置 pwd 后（与全局取并集），pwd 也会被拦截
    result = _validate_sensitive("SELECT name, pwd FROM employees", GLOBAL_SENSITIVE + ["pwd"])
    assert not result.is_safe
    assert "pwd" in result.blocked_detail
