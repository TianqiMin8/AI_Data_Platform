"""作业 5-2：动态上下文窗口。"""

from app.sql_engine.prompt_builder import build_text_to_sql_messages
from app.sql_engine.token_budget import (
    count_message_tokens,
    fit_history_to_budget,
    get_context_window,
    get_prompt_token_budget,
)

SCHEMA = [
    {
        "table_name": "demo_orders",
        "columns": [
            {"column_name": "id", "column_type": "INTEGER", "is_nullable": False},
            {"column_name": "category", "column_type": "VARCHAR(50)", "is_nullable": True},
            {"column_name": "amount", "column_type": "NUMERIC(10, 2)", "is_nullable": False},
        ],
    }
]


def _history(turns: int) -> list[dict]:
    """构造 N 轮历史，第 i 轮的内容带编号，方便断言保留了哪几轮。"""
    history = []
    for i in range(1, turns + 1):
        history.append({"role": "user", "content": f"第{i}轮问题：各品类销售额是多少？"})
        history.append({
            "role": "assistant",
            "content": f"SQL:SELECT category, SUM(amount) FROM demo_orders GROUP BY category -- turn {i}",
        })
    return history


def _kept_turn_numbers(messages: list[dict]) -> list[int]:
    return [
        int(m["content"].split("轮")[0].removeprefix("第"))
        for m in messages
        if m["role"] == "user" and m["content"].startswith("第")
    ]


# ── 预算 ──


def test_budget_is_70_percent_of_context_window():
    assert get_context_window("gpt-4o") == 128_000
    assert get_prompt_token_budget("gpt-4o") == 89_600


def test_unknown_model_uses_conservative_window():
    assert get_context_window("some-new-model") == 8_192


def test_token_count_grows_with_content():
    short = count_message_tokens([{"role": "user", "content": "总销售额"}])
    long = count_message_tokens([{"role": "user", "content": "总销售额" * 50}])
    assert long > short > 0


# ── 动态裁剪 ──


def test_large_budget_keeps_all_history():
    messages = build_text_to_sql_messages("哪个最高？", SCHEMA, conversation_history=_history(8))
    assert _kept_turn_numbers(messages) == list(range(1, 9))  # 不再固定只留 5 轮


def test_small_budget_keeps_only_newest_turns():
    # 用 2 轮 - 1 轮算每轮的 token：有历史时 system prompt 会多一段 Context Awareness Rules，
    # 用 1 轮 - 0 轮会把这段也算进"每轮"
    def tokens_with(turns: int) -> int:
        return count_message_tokens(
            build_text_to_sql_messages("哪个最高？", SCHEMA, conversation_history=_history(turns))
        )

    per_turn = tokens_with(2) - tokens_with(1)
    fixed = tokens_with(1) - per_turn

    # 预算只够装 3 轮多一点
    budget = fixed + per_turn * 3 + per_turn // 2
    messages = build_text_to_sql_messages(
        "哪个最高？", SCHEMA, conversation_history=_history(8), token_budget=budget
    )
    assert _kept_turn_numbers(messages) == [6, 7, 8]
    assert count_message_tokens(messages) <= budget


def test_budget_too_small_drops_all_history_but_keeps_question():
    messages = build_text_to_sql_messages(
        "哪个最高？", SCHEMA, conversation_history=_history(5), token_budget=10
    )
    assert _kept_turn_numbers(messages) == []
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "哪个最高？"}


def test_turns_are_dropped_whole():
    # 历史开头是一条孤立的 assistant 消息（例如上一轮的提问被截掉了），也要按整轮处理
    history = [{"role": "assistant", "content": "SQL:SELECT 1"}] + _history(2)
    kept, kept_turns = fit_history_to_budget(history, fixed_tokens=0, budget=10_000)
    assert kept_turns == 3
    assert kept == history

    # 预算只够最后一轮：保留的必须是完整的 user + assistant
    last_turn_tokens = count_message_tokens(history[-2:]) - 3
    kept, kept_turns = fit_history_to_budget(history, fixed_tokens=0, budget=last_turn_tokens)
    assert kept_turns == 1
    assert [m["role"] for m in kept] == ["user", "assistant"]


def test_stops_at_first_turn_that_does_not_fit():
    # 中间有一轮特别长：放不下它时，即使更早的轮次很短也不要，保持历史连续
    history = _history(1) + [
        {"role": "user", "content": "长问题"},
        {"role": "assistant", "content": "Result:" + "x" * 5000},
    ] + _history(1)
    kept, kept_turns = fit_history_to_budget(history, fixed_tokens=0, budget=200)
    assert kept_turns == 1
    assert kept == history[-2:]
