import re
from app.llm.factory import get_llm_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

TABLE_SELECTION_PROMPT = """Given the following list of database tables and a user's question, select the most relevant tables needed to answer the question.

Tables:
{table_list}

User's question:{question}

Return ONLY a JSON array of table names, nothing else. Example: ["orders", "products"]
Select at most{max_tables} tables."""


def keyword_select(
    question: str, schema_tables: list[dict], max_tables: int = 10
) -> list[dict]:
    """基于关键词匹配的快速表选择。"""

    question_lower = question.lower()
    question_words = set(re.findall(r'\w+', question_lower))

    scored_tables = []
    for table in schema_tables:
        score = 0
        table_name = table["table_name"].lower()

        # 表名匹配
        for word in table_name.split("_"):
            if word in question_words:
                score += 3

        # 列名匹配
        for col in table["columns"]:
            col_name = col["column_name"].lower()
            for word in col_name.split("_"):
                if word in question_words:
                    score += 1

            # 列注释匹配
            comment = (col.get("comment") or "").lower()
            if comment:
                for word in question_words:
                    if word in comment:
                        score += 2

        if score > 0:
            scored_tables.append((score, table))

    scored_tables.sort(key=lambda x: x[0], reverse=True)
    selected = [t for _, t in scored_tables[:max_tables]]

    logger.info(
        "table_selection.keyword",
        question=question[:50],
        total_tables=len(schema_tables),
        selected=len(selected),
    )

    return selected


async def llm_select(
    question: str, schema_tables: list[dict], max_tables: int = 10
) -> list[dict]:
    """基于 LLM 的精准表选择（备用方案）。"""

    table_list = "\n".join(
        f"-{t['table_name']}: columns = [{', '.join(c['column_name'] for c in t['columns'])}]"
        for t in schema_tables
    )

    prompt = TABLE_SELECTION_PROMPT.format(
        table_list=table_list,
        question=question,
        max_tables=max_tables,
    )

    adapter = get_llm_adapter()
    response = await adapter.generate([{"role": "user", "content": prompt}])

    try:
        import json
        selected_names = json.loads(response.strip())
        name_set = {n.lower() for n in selected_names}
        selected = [t for t in schema_tables if t["table_name"].lower() in name_set]

        logger.info(
            "table_selection.llm",
            question=question[:50],
            selected=[t["table_name"] for t in selected],
        )
        return selected

    except Exception as e:
        logger.warning("table_selection.llm_failed", error=str(e))
        return schema_tables[:max_tables]


async def select_relevant_tables(
    question: str,
    schema_tables: list[dict],
    max_tables: int = 10,
    keyword_min_count: int = 2,
) -> list[dict]:
    """组合策略：先关键词，不够再 LLM。"""

    if len(schema_tables) <= max_tables:
        return schema_tables  # 表少，不用选

    # 先试关键词匹配
    selected = keyword_select(question, schema_tables, max_tables)

    if len(selected) >= keyword_min_count:
        return selected

    # 关键词匹配不到足够的表，用 LLM
    return await llm_select(question, schema_tables, max_tables)