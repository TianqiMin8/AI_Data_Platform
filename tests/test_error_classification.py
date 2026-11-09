"""作业 3：异常类型分类 vs 字符串匹配分类。

在本地数据库上真实执行一批会出错的 SQL，拿到真实异常，分别交给两种分类器，对比准确率。
运行（-s 可以看到对比表）：
    poetry run pytest tests/test_error_classification.py -s
"""

import pytest
from sqlalchemy import create_engine, text

from app.config.settings import get_settings
from app.sql_engine.executor import _classify_error, _classify_error_by_message

SYNC_URL = get_settings().database_url.replace("+asyncpg", "")
BAD_PORT_URL = SYNC_URL.replace(":5432/", ":1/")

# (说明, 准备语句, 出错的 SQL, 期望的错误类型)
CASES = [
    ("关键字拼错", [], "SELEC category FROM demo_orders", "syntax_error"),
    ("表不存在", [], "SELECT * FROM demo_order", "undefined_table"),
    ("列不存在", [], "SELECT price FROM demo_orders", "undefined_column"),
    ("函数参数类型不对", [], "SELECT SUM(customer_name) FROM demo_orders", "undefined_function"),
    ("GROUP BY 漏列", [], "SELECT category, amount FROM demo_orders GROUP BY category", "grouping_error"),
    ("除以零（别名含 timeout）", [], "SELECT amount / 0 AS timeout_ratio FROM demo_orders", "division_by_zero"),
    ("值与类型不匹配", [], "SELECT * FROM demo_orders WHERE amount > 'abc'", "type_mismatch"),
    ("列名有歧义", [], "SELECT id FROM demo_orders a JOIN demo_orders b ON a.id = b.id", "ambiguous_column"),
    ("权限不足",
     ["CREATE ROLE hw3_no_access", "SET LOCAL ROLE hw3_no_access"],
     "SELECT * FROM demo_orders", "permission_denied"),
    ("执行超时", ["SET LOCAL statement_timeout = 100"], "SELECT pg_sleep(1)", "timeout"),
    ("连接失败", None, "SELECT 1", "connection_error"),  # None 表示连一个错误的端口
]

# 旧实现把"表不存在"和"列不存在"合并成一个 table_not_found，对比时按粗粒度算它对
OLD_TYPE_EQUIVALENTS = {
    "undefined_table": {"table_not_found"},
    "undefined_column": {"table_not_found"},
}


def _capture_exception(setup: list[str] | None, sql: str) -> Exception:
    """执行 SQL 并返回抛出的异常。所有语句都在事务里执行并回滚，不会改动数据库。"""
    url = BAD_PORT_URL if setup is None else SYNC_URL
    engine = create_engine(url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            with conn.begin() as tx:
                try:
                    for stmt in setup or []:
                        conn.execute(text(stmt))
                    conn.execute(text(sql))
                finally:
                    tx.rollback()
    except Exception as e:
        return e
    finally:
        engine.dispose()
    raise AssertionError(f"SQL 没有报错：{sql}")


def _old_is_correct(old_type: str, expected: str) -> bool:
    return old_type == expected or old_type in OLD_TYPE_EQUIVALENTS.get(expected, set())


@pytest.fixture(scope="module")
def captured():
    try:
        create_engine(SYNC_URL).connect().close()
    except Exception:
        pytest.skip("本地数据库不可用")
    return [(desc, sql, expected, _capture_exception(setup, sql)) for desc, setup, sql, expected in CASES]


def test_type_based_classification_is_correct(captured):
    for desc, sql, expected, exc in captured:
        assert _classify_error(exc, sql)["type"] == expected, desc


def test_accuracy_comparison(captured):
    rows = []
    new_correct = old_correct = 0
    for desc, sql, expected, exc in captured:
        new_type = _classify_error(exc, sql)["type"]
        old_type = _classify_error_by_message(str(exc).lower(), sql)["type"]
        new_ok = new_type == expected
        old_ok = _old_is_correct(old_type, expected)
        new_correct += new_ok
        old_correct += old_ok
        rows.append((desc, expected, old_type, "✓" if old_ok else "✗", new_type, "✓" if new_ok else "✗"))

    total = len(captured)
    print(f"\n{'场景':<22}{'期望':<20}{'字符串匹配':<24}{'异常类型':<22}")
    for desc, expected, old_type, old_mark, new_type, new_mark in rows:
        print(f"{desc:<20}{expected:<20}{old_type:<20}{old_mark:<4}{new_type:<20}{new_mark}")
    print(f"\n字符串匹配准确率：{old_correct}/{total} = {old_correct / total:.0%}")
    print(f"异常类型准确率：  {new_correct}/{total} = {new_correct / total:.0%}")

    assert new_correct == total
    assert new_correct > old_correct
