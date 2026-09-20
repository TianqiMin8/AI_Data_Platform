from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ConversationCreate(BaseModel):
    datasource_id: int
    title: str = "新对话"


class ConversationResponse(BaseModel):
    id: int
    datasource_id: Optional[int]
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class QueryResult(BaseModel):
    columns: list[str] = []
    rows: list[dict] = []
    row_count: int = 0
    total_row_count: int = 0
    execution_ms: float = 0.0
    truncated: bool = False


class ChatResponse(BaseModel):
    message_id: int
    role: str = "assistant"
    content: str
    generated_sql: Optional[str] = None
    query_result: Optional[QueryResult] = None
    blocked: bool = False                        # ← 新增
    block_reason: Optional[str] = None           # ← 新增
    model: str = ""
    usage: Optional[dict] = None



from datetime import datetime
from pydantic import BaseModel

class MessageHistoryItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    generated_sql: str | None = None
    execution_ms: float | None = None
    row_count: int | None = None


class MessageHistoryResponse(BaseModel):
    items: list[MessageHistoryItem]

from datetime import datetime
from pydantic import BaseModel


class ConversationListItem(BaseModel):
    id: int
    datasource_id: int | None
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_preview: str | None = None


class ConversationListResponse(BaseModel):
    items: list[ConversationListItem]
    next_cursor: int | None = None