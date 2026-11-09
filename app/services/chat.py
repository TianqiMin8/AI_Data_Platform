import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.datasource import DataSource
from app.models.conversation import Conversation, Message
from app.models.query_execution import QueryExecution
from app.schemas.chat import (
    ChatResponse,
    QueryResult,
    MessageHistoryItem,
    ConversationListItem,
)
from app.schemas.datasource import TableSchema
from app.services.datasource import get_datasource
from app.services.schema_inspector import introspect
from app.sql_engine.cache import get_cached_sql, set_cached_sql, invalidate_cache
from app.sql_engine.semantic_cache import find_similar_cached_sql, set_semantic_cached_sql
from app.sql_engine.cache_stats import record_cache_result
from app.sql_engine.generator import generate_sql
from app.sql_engine.validator import validate_sql
from app.sql_engine.executor import execute_sql
from app.core.errors import NotFoundError, SQLExecutionError, ValidationError
from app.config.settings import get_settings
from app.core.logging import get_logger
from app.sql_engine.table_selector import select_relevant_tables

logger = get_logger(__name__)


async def create_conversation(
    db: AsyncSession, user: User, datasource_id: int, title: str = "新对话"
) -> Conversation:
    ds = await get_datasource(db, user, datasource_id)

    conv = Conversation(
        user_id=user.id,
        datasource_id=ds.id,
        title=title,
    )
    db.add(conv)
    await db.flush()

    logger.info("conversation.created", conversation_id=conv.id, user_id=user.id)
    return conv


async def get_conversation(
    db: AsyncSession, user: User, conversation_id: int
) -> Conversation:
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise NotFoundError("对话", conversation_id)
    return conv

async def get_conversation_messages(
    db: AsyncSession,
    user: User,
    conversation_id: int,
) -> list[MessageHistoryItem]:
    """获取一个对话的所有消息，并附带 QueryExecution 信息。"""

    # 1. 先确认这个 conversation 属于当前用户
    conv = await get_conversation(db, user, conversation_id)

    # 2. 查询这个 conversation 下的所有消息
    # 同时 LEFT JOIN QueryExecution
    stmt = (
        select(Message, QueryExecution)
        .outerjoin(
            QueryExecution,
            QueryExecution.message_id == Message.id,
        )
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    # 3. 组装返回结果
    items = []

    for message, execution in rows:
        items.append(
            MessageHistoryItem(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                generated_sql=execution.generated_sql if execution else None,
                execution_ms=execution.execution_ms if execution else None,
                row_count=execution.row_count if execution else None,
            )
        )

    return items

async def send_message(
    db: AsyncSession, user: User, conversation_id: int, question: str
) -> ChatResponse:
    """核心方法：接收用户问题 → 生成 SQL → 执行 → 返回结果。"""

    # 1. 获取对话和数据源
    conv = await get_conversation(db, user, conversation_id)
    if not conv.datasource_id:
        raise ValidationError("该对话未关联数据源，无法生成 SQL")

    ds = await get_datasource(db, user, conv.datasource_id)

    # 2. 存用户消息
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()

    # 3. 获取 Schema（调用第二节课的自省能力）
    schema_result = await introspect(db, user, ds.id)
    schema_tables = [
        {"table_name": t.table_name, "columns": [c.model_dump() for c in t.columns]}
        for t in schema_result.tables
    ]
    

    # 3.3 Table Selection（大 Schema 优化）：生成 SQL 时只给 LLM 精选的表
    selected_tables = await select_relevant_tables(
        question=question,
        schema_tables=schema_tables,
        max_tables=10,
    )

    # ✅ 但 allowed_tables 仍然用全量表（安全校验不能因为 Table Selection 放松）
    allowed_table_names = [t["table_name"] for t in schema_tables]  # 全量
    allowed_columns = {
        t["table_name"]: [c["column_name"] for c in t["columns"]] for t in schema_tables
    }
    # 敏感列 = 全局配置 ∪ 该数据源的配置（数据源只能追加，不能去掉全局的）
    sensitive_columns = list(
        {*get_settings().sql_sensitive_columns, *(ds.sensitive_columns or [])}
    )

    # 3.5 构建多轮上下文
    conversation_history = await _build_conversation_history(db, conv.id)

    # 3.8 查缓存：先精确匹配（免费），没命中再语义匹配（一次 Embedding 调用）
    cached_sql = await get_cached_sql(ds.id, question)
    cache_source = "exact" if cached_sql else None
    question_embedding = None  # 语义查找时算出的 Embedding，写缓存时复用

    if not cached_sql:
        semantic_match, question_embedding = await find_similar_cached_sql(ds.id, question)
        if semantic_match and semantic_match.entry:
            cached_sql = semantic_match.entry.sql
            cache_source = "semantic"

    if cached_sql:
        # 缓存命中：跳过 LLM 调用，直接校验 + 执行
        sql = cached_sql
        llm_response = None

        validation = validate_sql(
            sql=sql,
            allowed_tables=allowed_table_names,
            allowed_columns=allowed_columns,
            sensitive_columns=sensitive_columns,
            db_type=ds.db_type,
        )
        if not validation.is_safe:
            # 缓存的 SQL 不再安全（可能 Schema 变了），清缓存走正常链路
            await invalidate_cache(ds.id)
            cached_sql = None
            cache_source = None

    # 命中率统计：缓存里拿到了 SQL 但没过校验的，算未命中（最终还是调了 LLM）
    await record_cache_result(cache_source)

    if not cached_sql:
        # 4. 正常 LLM 生成 SQL

        sql, llm_response = await generate_sql(
            question=question,
            schema_tables=selected_tables,  # ← 用精选的表
            db_type=ds.db_type,
            conversation_history=conversation_history,
        )

        # ✅ 4.5 新增：SQL 安全校验
        validation = validate_sql(
            sql=sql,
            allowed_tables=allowed_table_names,
            allowed_columns=allowed_columns,
            sensitive_columns=sensitive_columns,
            db_type=ds.db_type,
        )

        if not validation.is_safe:
            logger.warning(
                "sql.validation.blocked",
                level=validation.level,
                reason=validation.reason,
                sql=sql[:200],
            )

            # 存一条 assistant 消息告知用户
            block_message = f"抱歉，生成的查询未通过安全校验：{validation.reason}。请换一种方式描述您的问题。"
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=block_message,
            )
            db.add(assistant_msg)
            await db.flush()

            # 存 QueryExecution（记录被拦截的 SQL）
            qe = QueryExecution(
                message_id=assistant_msg.id,
                datasource_id=ds.id,
                generated_sql=sql,
                status="blocked",
                error_message=f"[{validation.level}]{validation.reason}:{validation.blocked_detail}",
            )
            db.add(qe)
            await db.flush()

            return ChatResponse(
                message_id=assistant_msg.id,
                content=block_message,
                generated_sql=sql,
                query_result=None,
                blocked=True,
                block_reason=validation.reason,
                model=llm_response.model,
                usage={
                    "prompt_tokens": llm_response.usage.prompt_tokens,
                    "completion_tokens": llm_response.usage.completion_tokens,
                },
            )

    # 5. 执行 SQL
    retry_count = 0
    try:
        exec_result = execute_sql(ds, sql)
    except SQLExecutionError as e:
        # ✅ 执行失败 → 尝试自修复（每轮：生成 → 校验 → 执行）
        logger.info("sql.execution_failed.attempting_retry", error=e.db_error[:200])

        from app.sql_engine.retry import retry_with_feedback

        retry_result = await retry_with_feedback(
            question=question,
            failed_sql=sql,
            error_message=e.db_error or e.message,  # 给 LLM 看数据库原始报错
            error_type=e.error_type,
            schema_tables=schema_tables,  # 用全量表：出错可能正是因为精选时漏了表
            datasource=ds,
            db_type=ds.db_type,
            conversation_history=conversation_history,
            allowed_tables=allowed_table_names,
            allowed_columns=allowed_columns,
            sensitive_columns=sensitive_columns,
        )
        retry_count = retry_result.retry_count
        if retry_result.llm_response:
            llm_response = retry_result.llm_response

        if not retry_result.success:
            # 重试用完仍失败：存 assistant 消息 + QueryExecution(status="failed")，正常返回而不是 500
            if cached_sql:
                await invalidate_cache(ds.id)  # 缓存里的 SQL 已经跑不通了

            fail_message = _build_retry_failed_message(
                retry_result.last_error_type, e.message, retry_result.retry_count
            )
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=fail_message,
            )
            db.add(assistant_msg)
            await db.flush()

            qe = QueryExecution(
                message_id=assistant_msg.id,
                datasource_id=ds.id,
                generated_sql=retry_result.sql,
                status="failed",
                error_message=retry_result.last_error[:2000],
                retry_count=retry_count,
            )
            db.add(qe)
            await db.flush()

            return ChatResponse(
                message_id=assistant_msg.id,
                content=fail_message,
                generated_sql=retry_result.sql,
                query_result=None,
                model=llm_response.model if llm_response else "",
            )

        sql = retry_result.sql
        exec_result = retry_result.exec_result

    # ✅ 执行成功后才写缓存（缓存的是最终能跑通的 SQL，包括重试修正后的）
    if not cached_sql or retry_count:
        await set_cached_sql(ds.id, question, sql)
        await set_semantic_cached_sql(ds.id, question, sql, question_embedding)

    # ✅ 6. 升级：用 LLM 生成自然语言解释
    from app.sql_engine.explainer import explain_result

    answer = await explain_result(
        question=question,
        sql=sql,
        rows=exec_result.rows,
        row_count=exec_result.row_count,
    )

    # 7. 存 assistant 消息
    assistant_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    await db.flush()

    # 8. 存 QueryExecution
    qe = QueryExecution(
        message_id=assistant_msg.id,
        datasource_id=ds.id,
        generated_sql=sql,
        status="success",
        result_summary=json.dumps(exec_result.rows[:5], ensure_ascii=False, default=str),
        row_count=exec_result.row_count,
        execution_ms=exec_result.execution_ms,
        retry_count=retry_count,
    )
    db.add(qe)
    await db.flush()

    # 缓存命中时没有调用 LLM，llm_response 为 None
    model = llm_response.model if llm_response else "cache"
    usage = (
        {
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "completion_tokens": llm_response.usage.completion_tokens,
        }
        if llm_response
        else None
    )

    logger.info(
        "chat.completed",
        conversation_id=conv.id,
        sql_length=len(sql),
        row_count=exec_result.row_count,
        execution_ms=exec_result.execution_ms,
        model=model,
    )

    return ChatResponse(
        message_id=assistant_msg.id,
        content=answer,
        generated_sql=sql,
        query_result=QueryResult(**exec_result.to_dict()),
        model=model,
        usage=usage,
    )


def _format_answer(question: str, sql: str, result: dict) -> str:
    """把 SQL 执行结果格式化为自然语言回答。"""
    rows = result["rows"]
    if not rows:
        return f"查询已执行，但没有返回结果。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    if len(rows) == 1 and len(rows[0]) == 1:
        key = list(rows[0].keys())[0]
        value = rows[0][key]
        return f"查询结果为 **{value}**。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    row_count = result["row_count"]
    truncated = result.get("truncated", False)
    summary = f"查询返回了{row_count} 条结果"
    if truncated:
        summary += f"（已截取前{row_count} 条）"
    summary += f"。\n\n执行的 SQL：\n```sql\n{sql}\n```"

    return summary

async def list_conversations(
    db: AsyncSession,
    user: User,
    cursor: int | None = None,
    limit: int = 20,
) -> tuple[list[ConversationListItem], int | None]:

    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(
            Conversation.updated_at.desc(),
            Conversation.id.desc(),
        )
    )

    if cursor is not None:
        stmt = stmt.where(Conversation.id < cursor)

    stmt = stmt.limit(limit + 1)

    result = await db.execute(stmt)
    conversations = list(result.scalars().all())

    next_cursor = None

    if len(conversations) > limit:
        conversations = conversations[:limit]
        next_cursor = conversations[-1].id

    items = []

    for conv in conversations:
        last_msg_stmt = (
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(
                Message.created_at.desc(),
                Message.id.desc(),
            )
            .limit(1)
        )

        last_msg_result = await db.execute(last_msg_stmt)
        last_message = last_msg_result.scalar_one_or_none()

        preview = None

        if last_message:
            preview = last_message.content[:100]

        items.append(
            ConversationListItem(
                id=conv.id,
                datasource_id=conv.datasource_id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                last_message_preview=preview,
            )
        )

    return items, next_cursor

# app/services/chat.py 新增辅助函数

_RETRY_FAILED_HINTS = {
    "syntax_error": "可以换一种更直白的说法，或者把问题拆成几个简单的小问题。",
    "undefined_table": "请确认问题里提到的数据在当前数据源中存在，必要时可以先问「有哪些表」。",
    "undefined_column": "请确认问题里提到的字段存在，必要时可以先问「某张表有哪些字段」。",
    "undefined_function": "问题里的计算方式可能不适用于这类数据（例如对文字求和），请换一种统计方式。",
    "grouping_error": "请更明确地说明按什么维度分组统计，例如「按品类统计销售额」。",
    "type_mismatch": "请检查问题中的条件格式，例如日期写成「2026年3月」、金额写成数字。",
    "table_not_found": "请确认问题里提到的数据在当前数据源中存在，必要时可以先问「有哪些表」「某张表有哪些字段」。",
    "validation_failed": "生成的查询没有通过安全校验，请只询问允许访问的数据。",
    "timeout": "查询数据量可能过大，请缩小范围，例如加上时间条件或限制条数。",
    "permission_denied": "当前数据源的数据库账号没有查询权限，请联系管理员授予 SELECT 权限。",
    "connection_error": "暂时无法连接到数据库，请检查数据源配置或稍后重试。",
}


def _build_retry_failed_message(error_type: str, user_message: str, retry_count: int) -> str:
    """查询最终失败时给用户的提示：说明发生了什么 + 针对错误类型的具体建议。"""
    hint = _RETRY_FAILED_HINTS.get(error_type, "请换一种方式描述您的问题，或稍后重试。")
    if retry_count == 0:
        # 不可修复的错误没有重试，不能说"尝试修正了几次"
        return f"抱歉，查询没有成功（{user_message}）。这个问题无法通过调整查询来解决。建议：{hint}"
    return f"抱歉，这个问题我尝试自动修正了 {retry_count} 次查询，仍然没有成功（{user_message}）。建议：{hint}"


async def _build_conversation_history(
    db: AsyncSession, conversation_id: int, max_turns: int | None = None
) -> list[dict]:
    """从数据库读取最近 N 轮对话，构建 LLM messages 格式的历史。

    这里的 N 只是读取上限（默认 max_history_fetch_turns=20），避免长对话一次读太多；
    最终保留几轮由 prompt_builder 按 token 预算动态决定。
    """
    max_turns = max_turns or get_settings().max_history_fetch_turns

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        # 同一请求里 user 和 assistant 消息在同一事务写入，created_at（事务开始时间）相同，
        # 必须再按自增 id 排序，否则同一轮内可能出现"先回答后提问"
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(max_turns * 2 + 2)  # 多取一些，后面截断
    )
    result = await db.execute(stmt)
    messages = list(reversed(result.scalars().all()))

    # 排除最后一条（当前用户刚发的消息）
    if messages and messages[-1].role == "user":
        messages = messages[:-1]

    history = []
    for msg in messages:
        if msg.role == "user":
            history.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            # 尝试从 QueryExecution 获取 SQL，拼成更精确的上下文
            qe = await _get_query_execution_for_message(db, msg.id)
            if qe and qe.generated_sql:
                content = f"SQL:{qe.generated_sql}"
                if qe.result_summary:
                    content += f"\nResult:{qe.result_summary}"
            else:
                content = msg.content[:500]
            history.append({"role": "assistant", "content": content})

    return history[-max_turns * 2:]  # 只保留最近 N 轮


async def _get_query_execution_for_message(
    db: AsyncSession, message_id: int
) -> QueryExecution | None:
    stmt = select(QueryExecution).where(QueryExecution.message_id == message_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
