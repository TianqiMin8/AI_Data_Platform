from dataclasses import dataclass

from app.llm.factory import get_llm_adapter
from app.llm.base import LLMResponse
from app.models.datasource import DataSource
from app.sql_engine.prompt_builder import build_text_to_sql_messages
from app.sql_engine.validator import validate_sql
from app.sql_engine.executor import execute_sql, is_retryable, SQLExecutionResult
from app.core.errors import SQLExecutionError
from app.core.logging import get_logger

logger = get_logger(__name__)

RETRY_PROMPT = """The previous SQL query failed with the following error:

Error:{error_message}

Failed SQL:
{failed_sql}

Please generate a corrected SQL query that fixes this error.
Remember:
1. ONLY generate SELECT statements
2. ONLY use table and column names from the provided schema
3. Output ONLY the raw SQL query
4. Fix the specific error mentioned above"""


@dataclass
class RetryResult:
    """一次自修复流程的结果。"""
    success: bool
    retry_count: int                                # 实际调用 LLM 重新生成的次数
    sql: str                                        # 成功时是可用的 SQL；失败时是最后一次尝试的 SQL
    llm_response: LLMResponse | None = None         # 最后一次 LLM 调用的响应
    exec_result: SQLExecutionResult | None = None   # 成功时的执行结果
    last_error: str = ""                            # 失败时最后一次的错误（给 LLM 看的原始报错）
    last_error_type: str = ""                       # 失败时最后一次的错误分类


async def retry_with_feedback(
    question: str,
    failed_sql: str,
    error_message: str,
    schema_tables: list[dict],
    datasource: DataSource,
    db_type: str = "postgresql",
    conversation_history: list[dict] | None = None,
    max_retries: int = 2,
    allowed_tables: list[str] | None = None,
    allowed_columns: dict[str, list[str]] | None = None,
    sensitive_columns: list[str] | None = None,
    error_type: str = "unknown",
) -> RetryResult:
    """SQL 执行失败后，把错误反馈给 LLM 重新生成，最多重试 max_retries 次。

    每一轮：生成 → 安全校验 → 执行。任一步失败，就把这一步的错误作为下一轮的反馈。
    不可修复的错误（连接失败、权限不足等）直接返回，不调用 LLM。
    """

    from app.sql_engine.generator import clean_sql_output

    current_error = error_message
    current_error_type = error_type
    current_failed_sql = failed_sql
    response: LLMResponse | None = None

    if not is_retryable(error_type):
        logger.info("sql.retry.skipped_unrecoverable", error_type=error_type)
        return RetryResult(
            success=False,
            retry_count=0,
            sql=failed_sql,
            last_error=error_message,
            last_error_type=error_type,
        )

    for attempt in range(1, max_retries + 1):
        logger.info(
            "sql.retry.attempt",
            attempt=attempt,
            max_retries=max_retries,
            error=current_error[:100],
        )

        # 构建修正 prompt
        retry_content = RETRY_PROMPT.format(
            error_message=current_error,
            failed_sql=current_failed_sql,
        )

        messages = build_text_to_sql_messages(
            question=question,
            schema_tables=schema_tables,
            db_type=db_type,
            conversation_history=conversation_history,
        )

        # 把失败信息作为额外上下文追加
        messages.append({"role": "assistant", "content": current_failed_sql})
        messages.append({"role": "user", "content": retry_content})

        adapter = get_llm_adapter()
        response = await adapter.generate_with_usage(messages)
        new_sql = clean_sql_output(response.content)

        if not new_sql:
            logger.warning("sql.retry.empty_result", attempt=attempt)
            current_error = "The response did not contain a SQL query."
            current_error_type = "empty_sql"
            continue

        # 修正后的 SQL 也要过安全校验
        validation = validate_sql(
            sql=new_sql,
            allowed_tables=allowed_tables,
            allowed_columns=allowed_columns,
            sensitive_columns=sensitive_columns,
            db_type=db_type,
        )

        if not validation.is_safe:
            logger.warning(
                "sql.retry.validation_failed",
                attempt=attempt,
                reason=validation.reason,
            )
            # 校验失败的原因也反馈给 LLM，下一轮针对性修正
            current_failed_sql = new_sql
            current_error = f"Security validation failed: {validation.reason} ({validation.blocked_detail})"
            current_error_type = "validation_failed"
            continue

        # 校验通过 → 执行；执行失败就带着数据库原始报错进入下一轮
        try:
            exec_result = execute_sql(datasource, new_sql)
        except SQLExecutionError as e:
            logger.warning("sql.retry.execution_failed", attempt=attempt, error=e.db_error[:100])
            current_failed_sql = new_sql
            current_error = e.db_error or e.message
            current_error_type = e.error_type
            if not is_retryable(e.error_type):
                # 例如重试过程中数据库断开了：剩下的重试次数也没用
                logger.info("sql.retry.stopped_unrecoverable", attempt=attempt, error_type=e.error_type)
                return RetryResult(
                    success=False,
                    retry_count=attempt,
                    sql=current_failed_sql,
                    llm_response=response,
                    last_error=current_error,
                    last_error_type=current_error_type,
                )
            continue

        logger.info("sql.retry.success", attempt=attempt, new_sql_length=len(new_sql))
        return RetryResult(
            success=True,
            retry_count=attempt,
            sql=new_sql,
            llm_response=response,
            exec_result=exec_result,
        )

    logger.warning("sql.retry.exhausted", max_retries=max_retries)
    return RetryResult(
        success=False,
        retry_count=max_retries,
        sql=current_failed_sql,
        llm_response=response,
        last_error=current_error,
        last_error_type=current_error_type,
    )
