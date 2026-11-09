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
from app.sql_engine.cache import invalidate_cache
from app.sql_engine.generator import generate_sql
from app.sql_engine.executor import execute_sql
from app.core.errors import NotFoundError, SQLExecutionError, ValidationError
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
    

    # 3.3 Table Selection（大 Schema 优化）
    selected_tables = await select_relevant_tables(
        question=question,
        schema_tables=schema_tables,
        max_tables=10,
    )

    # 后续步骤使用 selected_tables 替代 schema_tables
    sql, llm_response = await generate_sql(
        question=question,
        schema_tables=selected_tables,  # ← 用精选的表
        db_type=ds.db_type,
        conversation_history=conversation_history,
    )

    # ✅ 但 allowed_tables 仍然用全量表（安全校验不能因为 Table Selection 放松）
    allowed_table_names = [t["table_name"] for t in schema_tables]  # 全量
    
    # 3.5 构建多轮上下文
    conversation_history = await _build_conversation_history(db, conv.id)

    # app/services/chat.py send_message 中，步骤 4 之前新增缓存查询

    from app.sql_engine.cache import get_cached_sql, set_cached_sql

    # 3.8 查缓存
    cached_sql = await get_cached_sql(ds.id, question)

    if cached_sql:
        # 缓存命中：跳过 LLM 调用，直接校验 + 执行
        sql = cached_sql
        llm_response = None

        validation = validate_sql(sql=sql, allowed_tables=allowed_table_names, db_type=ds.db_type)
        if not validation.is_safe:
            # 缓存的 SQL 不再安全（可能 Schema 变了），清缓存走正常链路
            await invalidate_cache(ds.id)
            cached_sql = None

    if not cached_sql:
        # 4. 正常 LLM 生成 SQL

        sql, llm_response = await generate_sql(
            question=question,
            schema_tables=schema_tables,
            db_type=ds.db_type,
            conversation_history = conversation_history
        )

        # ✅ 4.5 新增：SQL 安全校验
        from app.sql_engine.validator import validate_sql, ValidationResult

        allowed_table_names = [t["table_name"] for t in schema_tables]
        validation = validate_sql(
            sql=sql,
            allowed_tables=allowed_table_names,
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
                model=llm_response.model,
                usage={
                    "prompt_tokens": llm_response.usage.prompt_tokens,
                    "completion_tokens": llm_response.usage.completion_tokens,
                },
            )
        
        # ✅ 生成成功后写入缓存
        await set_cached_sql(ds.id, question, sql)

    # 5. 执行 SQL
    try:
        exec_result = execute_sql(ds, sql)
    except SQLExecutionError as e:
        # ✅ 执行失败 → 尝试自修复
        logger.info("sql.execution_failed.attempting_retry", error=str(e))

        from app.sql_engine.retry import retry_with_feedback

        retry_result = await retry_with_feedback(
            question=question,
            failed_sql=sql,
            error_message=str(e),
            schema_tables=schema_tables,
            db_type=ds.db_type,
            conversation_history=conversation_history,
            allowed_tables=allowed_table_names,
        )

        if retry_result:
            sql, llm_response = retry_result
            exec_result = execute_sql(ds, sql)  # 用修正后的 SQL 重新执行
        else:
            # 重试也失败了，返回错误信息
            # error_msg = f"查询执行失败，自动修正也未成功：{str(e)}"
            # ... 存 assistant 消息 + QueryExecution(status="failed") + 返回 ...
            raise
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
        result_summary=json.dumps(exec_result["rows"][:5], ensure_ascii=False, default=str),
        row_count=exec_result["row_count"],
        execution_ms=exec_result["execution_ms"],
    )
    db.add(qe)
    await db.flush()

    logger.info(
        "chat.completed",
        conversation_id=conv.id,
        sql_length=len(sql),
        row_count=exec_result["row_count"],
        execution_ms=exec_result["execution_ms"],
        model=llm_response.model,
    )

    return ChatResponse(
        message_id=assistant_msg.id,
        content=answer,
        generated_sql=sql,
        query_result=QueryResult(**exec_result),
        model=llm_response.model,
        usage={
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "completion_tokens": llm_response.usage.completion_tokens,
        },
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

async def _build_conversation_history(
    db: AsyncSession, conversation_id: int, max_turns: int = 5
) -> list[dict]:
    """从数据库读取最近 N 轮对话，构建 LLM messages 格式的历史。"""

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
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
