"""作业 5-3：可修复 vs 不可修复错误。用假的 LLM 和执行器，统计 LLM 被调用了几次。"""

import asyncio
from types import SimpleNamespace

import pytest

import app.sql_engine.retry as retry_module
from app.core.errors import SQLExecutionError
from app.sql_engine.executor import ERROR_RECOVERABILITY, PG_ERROR_TYPES, is_retryable
from app.sql_engine.retry import retry_with_feedback

SCHEMA = [
    {
        "table_name": "demo_orders",
        "columns": [
            {"column_name": "category", "column_type": "VARCHAR(50)", "is_nullable": True},
            {"column_name": "amount", "column_type": "NUMERIC(10, 2)", "is_nullable": False},
        ],
    }
]
GOOD_SQL = "SELECT category, SUM(amount) FROM demo_orders GROUP BY category"


class FakeAdapter:
    def __init__(self, sql: str = GOOD_SQL):
        self.sql = sql
        self.calls = 0

    async def generate_with_usage(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.sql, model="fake")


@pytest.fixture
def fake_llm(monkeypatch):
    adapter = FakeAdapter()
    monkeypatch.setattr(retry_module, "get_llm_adapter", lambda: adapter)
    return adapter


def _fake_executor(monkeypatch, outcomes: list):
    """依次返回 outcomes：Exception 就抛出，否则当作执行结果返回。"""
    remaining = list(outcomes)

    def execute(datasource, sql):
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(retry_module, "execute_sql", execute)


def _run(error_type: str, error_message: str = "boom"):
    return asyncio.run(
        retry_with_feedback(
            question="各品类销售额",
            failed_sql="SELECT ...",
            error_message=error_message,
            error_type=error_type,
            schema_tables=SCHEMA,
            datasource=SimpleNamespace(id=1),
        )
    )


# ── 分类表 ──


def test_at_least_five_error_types_on_each_side():
    recoverable = [t for t, ok in ERROR_RECOVERABILITY.items() if ok]
    unrecoverable = [t for t, ok in ERROR_RECOVERABILITY.items() if not ok]
    assert len(recoverable) >= 5
    assert len(unrecoverable) >= 3


def test_every_classified_error_type_has_recoverability():
    # executor 能分出来的每一种类型，都必须明确标了能不能重试
    for _, error_type, _ in PG_ERROR_TYPES:
        assert error_type in ERROR_RECOVERABILITY, error_type


def test_unregistered_type_is_not_retryable():
    assert not is_retryable("something_new")


# ── 不可修复：不调用 LLM ──


@pytest.mark.parametrize("error_type", ["connection_error", "permission_denied", "timeout", "unknown"])
def test_unrecoverable_error_skips_llm(fake_llm, error_type):
    result = _run(error_type, "permission denied for table demo_orders")
    assert fake_llm.calls == 0
    assert not result.success
    assert result.retry_count == 0
    assert result.last_error_type == error_type
    assert result.last_error == "permission denied for table demo_orders"


# ── 可修复：正常重试 ──


@pytest.mark.parametrize("error_type", ["syntax_error", "undefined_column", "undefined_function", "grouping_error"])
def test_recoverable_error_retries_and_succeeds(fake_llm, monkeypatch, error_type):
    _fake_executor(monkeypatch, ["<rows>"])
    result = _run(error_type)
    assert fake_llm.calls == 1
    assert result.success
    assert result.retry_count == 1
    assert result.sql == GOOD_SQL


def test_recoverable_error_uses_all_retries_when_still_failing(fake_llm, monkeypatch):
    still_wrong = SQLExecutionError("语法错误", db_error="syntax error", error_type="syntax_error")
    _fake_executor(monkeypatch, [still_wrong, still_wrong])
    result = _run("syntax_error")
    assert fake_llm.calls == 2
    assert result.retry_count == 2
    assert not result.success


def test_stops_when_retry_hits_unrecoverable_error(fake_llm, monkeypatch):
    # 第 1 次重试执行时数据库断开：不再浪费第 2 次重试
    lost = SQLExecutionError("连不上", db_error="connection refused", error_type="connection_error")
    _fake_executor(monkeypatch, [lost])
    result = _run("syntax_error")
    assert fake_llm.calls == 1
    assert result.retry_count == 1
    assert result.last_error_type == "connection_error"
