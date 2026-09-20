from app.llm.factory import get_llm_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

EXPLAIN_PROMPT = """You are a data analyst assistant. The user asked a data question, and the system executed a SQL query and got the result.

Your task: Explain the query result in natural language. Be concise and helpful.

Rules:
1. Directly answer the user's original question based on the result
2. Mention key numbers and highlight important findings
3. If the result has multiple rows, summarize the pattern or highlight the top/bottom entries
4. If the result is empty, explain what that means in the context of the question
5. Keep the response under 100 words
6. Use the same language as the user's question (if question is in Chinese, answer in Chinese)

User's question:{question}

SQL executed:
{sql}

Result ({row_count} rows):
{result_text}"""


async def explain_result(
    question: str,
    sql: str,
    rows: list[dict],
    row_count: int,
) -> str:
    """用 LLM 将 SQL 执行结果翻译成自然语言。"""

    # 简单结果直接用模版（省一次 LLM 调用）
    if row_count == 0:
        return f"查询已执行，但没有返回结果。可能是条件过滤掉了所有数据。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    if row_count == 1 and len(rows[0]) == 1:
        key = list(rows[0].keys())[0]
        value = rows[0][key]
        return f"查询结果为 **{value}**。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    # 多行结果用 LLM 解释
    result_preview = rows[:10]
    result_text = _format_rows_for_prompt(result_preview)

    prompt = EXPLAIN_PROMPT.format(
        question=question,
        sql=sql,
        row_count=row_count,
        result_text=result_text,
    )

    try:
        adapter = get_llm_adapter()
        explanation = await adapter.generate([
            {"role": "user", "content": prompt}
        ])

        explanation += f"\n\n执行的 SQL：\n```sql\n{sql}\n```"

        logger.info("sql.result.explained", question=question[:50], row_count=row_count)
        return explanation

    except Exception as e:
        # LLM 解释失败不应该阻断主链路，降级到简单模版
        logger.warning("sql.result.explain_failed", error=str(e))
        return f"查询返回了{row_count} 条结果。\n\n执行的 SQL：\n```sql\n{sql}\n```"


def _format_rows_for_prompt(rows: list[dict]) -> str:
    """将行数据格式化为 LLM 可读的文本。"""
    if not rows:
        return "(empty)"
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        line = " | ".join(str(row.get(h, "")) for h in headers)
        lines.append(line)
    return "\n".join(lines)