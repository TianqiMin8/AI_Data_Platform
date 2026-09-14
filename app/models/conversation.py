from sqlalchemy import String, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional

from app.models.base import Base, TimestampMixin


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    datasource_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("datasources.id"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(500), default="新对话")

    # 关系
    user: Mapped["User"] = relationship(back_populates="conversations")
    datasource: Mapped[Optional["DataSource"]] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", order_by="Message.created_at"
    )


class Message(TimestampMixin, Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user / assistant / system
    content: Mapped[str] = mapped_column(String(10000))

    # 关系
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    query_execution: Mapped[Optional["QueryExecution"]] = relationship(
        back_populates="message", uselist=False
    )