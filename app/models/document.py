from sqlalchemy import String, Integer, ForeignKey, Text, Float, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional

from app.models.base import Base, TimestampMixin


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(20))  # pdf / txt / md / docx
    file_size: Mapped[int] = mapped_column(Integer)      # bytes
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending → processing → completed → failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    total_characters: Mapped[int] = mapped_column(Integer, default=0)
    # 文件内容的 SHA-256，用于上传去重。旧数据没有该值，所以允许为空
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # 去重查询是"某用户 + 某哈希"，建联合索引。不设唯一约束：失败的文档会保留，重新上传时会再建一条
    __table_args__ = (Index("ix_documents_user_id_content_hash", "user_id", "content_hash"),)

    # 关系
    owner: Mapped["User"] = relationship()
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


# app/models/document.py 修改 Chunk 模型

from pgvector.sqlalchemy import Vector


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # ✅ Embedding 向量（1536 维，对应 text-embedding-3-small）
    embedding = mapped_column(Vector(1536), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # 关系
    document: Mapped["Document"] = relationship(back_populates="chunks")
    