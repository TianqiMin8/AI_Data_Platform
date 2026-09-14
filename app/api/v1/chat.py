from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.chat import (
    ConversationCreate,
    ConversationResponse,
    ChatRequest,
    ChatResponse,
)
from app.services import chat as chat_service

router = APIRouter(prefix="/conversations", tags=["chat"])


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    req: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await chat_service.create_conversation(
        db, current_user, req.datasource_id, req.title
    )
    return conv


@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def send_chat_message(
    conversation_id: int,
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await chat_service.send_message(
        db, current_user, conversation_id, req.message
    )