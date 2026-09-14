from datetime import datetime

from app.core.logging import get_logger

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


def build_text_to_sql_messages(
    question: str,
    schema_tables: list[dict],
    db_type: str = "postgresql",
    include_few_shot: bool = True,
    max_tables: int = 30,
) -> list[dict]:
    """构建发给 LLM 的完整 messages 列表。"""

    # 大 Schema 场景：限制表数量
    if len(schema_tables) > max_tables:
        logger.warning(
            "prompt.schema_truncated",
            total_tables=len(schema_tables),
            max_tables=max_tables,
        )
        schema_tables = schema_tables[:max_tables]

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_content = SYSTEM_PROMPT.format(current_date=current_date)
    system_content += f"\n\nSQL Dialect:{db_type.upper()}"

    # Schema Context
    schema_text = format_schema_context(schema_tables)
    system_content += f"\n\nDatabase Schema:\n{schema_text}"

    messages = [{"role": "system", "content": system_content}]

    # Few-shot 示例
    if include_few_shot:
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["sql"]})

    # 用户实际问题
    messages.append({"role": "user", "content": question})

    logger.info(
        "prompt.built",
        question=question,
        schema_tables=len(schema_tables),
        message_count=len(messages),
    )

    return messages