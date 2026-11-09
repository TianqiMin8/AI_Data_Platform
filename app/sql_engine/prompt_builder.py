from datetime import datetime

from app.core.logging import get_logger
from app.sql_engine.token_budget import (
    count_message_tokens,
    fit_history_to_budget,
    get_prompt_token_budget,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a SQL generation assistant for a data analytics platform.

Your task: Given a database schema and a natural language question, generate a single executable SQL query.

Rules:
1. ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, or any DDL/DML.
2. ONLY use table names and column names that appear in the provided schema.
3. Output ONLY the raw SQL query — no explanations, no markdown code blocks, no comments.
4. Use the correct SQL dialect as specified.
5. If the question is ambiguous, make reasonable assumptions and generate the most likely intended query.
6. Use table aliases for readability when joining multiple tables.
7. Always consider NULL values in aggregations.
8. For date/time questions, assume the current date is{current_date}."""

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the total revenue this month?",
        "sql": "SELECT SUM(amount) AS total_revenue FROM orders WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) AND created_at < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'",
    },
    {
        "question": "Show me the top 5 customers by order count",
        "sql": "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 5",
    },
]


def format_schema_context(tables: list[dict]) -> str:
    """将 Schema 自省结果格式化为 Prompt 可用的文本。"""
    lines = []
    for table in tables:
        table_name = table["table_name"]
        lines.append(f"Table:{table_name}")
        for col in table["columns"]:
            nullable = "NULLABLE" if col["is_nullable"] else "NOT NULL"
            comment = f" —{col['comment']}" if col.get("comment") else ""
            lines.append(f"  -{col['column_name']} ({col['column_type']},{nullable}){comment}")
        lines.append("")
    return "\n".join(lines)


# 在 build_text_to_sql_messages 函数中新增 conversation_history 参数

def build_text_to_sql_messages(
    question: str,
    schema_tables: list[dict],
    db_type: str = "postgresql",
    include_few_shot: bool = True,
    max_tables: int = 30,
    conversation_history: list[dict] | None = None,
    token_budget: int | None = None,
) -> list[dict]:
    """构建发给 LLM 的完整 messages 列表，支持多轮对话上下文。

    历史轮数是动态的：先算出 system / Schema / few-shot / 当前问题这些必须保留部分的 token，
    剩下的预算（默认 context window 的 70%）从最新一轮往前装历史，装不下就丢掉更早的。
    token_budget: 覆盖默认预算，主要用于测试。
    """

    if len(schema_tables) > max_tables:
        logger.warning("prompt.schema_truncated", total=len(schema_tables), max=max_tables)
        schema_tables = schema_tables[:max_tables]

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = SYSTEM_PROMPT.format(current_date=current_date)
    system_content += f"\n\nSQL Dialect:{db_type.upper()}"

    schema_text = format_schema_context(schema_tables)
    system_content += f"\n\nDatabase Schema:\n{schema_text}"

    # 多轮对话时追加上下文指令
    if conversation_history:
        system_content += """

Context Awareness Rules:
- The user may refer to previous questions and results using pronouns like "it", "that", "the same"
- When the user says "show me more detail" or "break it down", they mean the previous query's result
- When the user adds a filter like "only Q1" or "just this year", apply it to the previous query pattern
- Always consider the full conversation context when generating SQL"""

    prefix = [{"role": "system", "content": system_content}]

    # Few-shot 示例
    if include_few_shot:
        for example in FEW_SHOT_EXAMPLES:
            prefix.append({"role": "user", "content": example["question"]})
            prefix.append({"role": "assistant", "content": example["sql"]})

    current_question = {"role": "user", "content": question}

    # ✅ 多轮历史上下文：按 token 预算动态决定保留几轮
    budget = token_budget or get_prompt_token_budget()
    fixed_tokens = count_message_tokens(prefix + [current_question])
    history, kept_turns = fit_history_to_budget(
        conversation_history or [], fixed_tokens=fixed_tokens, budget=budget
    )
    if fixed_tokens > budget:
        logger.warning("prompt.over_budget_without_history", fixed_tokens=fixed_tokens, budget=budget)

    messages = prefix + history + [current_question]

    logger.info(
        "prompt.built",
        question=question,
        schema_tables=len(schema_tables),
        history_turns=kept_turns,
        history_turns_available=len([m for m in conversation_history or [] if m["role"] == "user"]),
        message_count=len(messages),
        estimated_prompt_tokens=count_message_tokens(messages),
        token_budget=budget,
    )

    return messages