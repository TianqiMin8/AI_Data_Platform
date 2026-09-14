from sqlalchemy import String, Integer, ForeignKey, Text, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional

from app.models.base import Base, TimestampMixin


class QueryExecution(TimestampMixin, Base):
    __tablename__ = "query_executions"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id"), unique=True, index=True
    )
    datasource_id: Mapped[int] = mapped_column(
        ForeignKey("datasources.id"), index=True
    )
    generated_sql: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # status: pending → running → success / failed
    result_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    execution_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # 关系
    message: Mapped["Message"] = relationship(back_populates="query_execution")