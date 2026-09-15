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
from app.sql_engine.generator import generate_sql
from app.sql_engine.executor import execute_sql
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger

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

    # 4. 生成 SQL
    sql, llm_response = await generate_sql(
        question=question,
        schema_tables=schema_tables,
        db_type=ds.db_type,
    )

    # 5. 执行 SQL
    exec_result = execute_sql(ds, sql)

    # 6. 构造自然语言回答
    answer = _format_answer(question, sql, exec_result)

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