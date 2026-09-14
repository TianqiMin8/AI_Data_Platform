import re

from app.llm.factory import get_llm_adapter
from app.llm.base import LLMResponse
from app.sql_engine.prompt_builder import build_text_to_sql_messages
from app.core.logging import get_logger
from app.core.errors import SQLGenerationError

logger = get_logger(__name__)


def clean_sql_output(raw: str) -> str:
    """从 LLM 输出中提取纯 SQL。"""
    text = raw.strip()

    # 去掉 markdown 代码块
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 去掉末尾分号（有些数据库驱动不需要）
    text = text.rstrip(";").strip()

    # 去掉开头可能的 "SQL:" 之类的标记
    text = re.sub(r"^(?:SQL|Query)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    return text


async def generate_sql(
    question: str,
    schema_tables: list[dict],
    db_type: str = "postgresql",
) -> tuple[str, LLMResponse]:
    """生成 SQL，返回 (清洗后的SQL, LLM完整响应)。"""

    messages = build_text_to_sql_messages(
        question=question,
        schema_tables=schema_tables,
        db_type=db_type,
    )

    adapter = get_llm_adapter()
    response = await adapter.generate_with_usage(messages)

    sql = clean_sql_output(response.content)

    if not sql:
        raise SQLGenerationError("LLM 未返回有效的 SQL 语句")

    logger.info(
        "sql.generated",
        question=question,
        sql_length=len(sql),
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
    )

    return sql, response