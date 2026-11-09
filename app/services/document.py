import hashlib
import os
import tempfile
from fastapi import UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.document import Document, Chunk
from app.rag.pipeline import process_document
from app.config.settings import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_TYPES = {"pdf", "txt", "md", "markdown", "docx", "html", "htm"}


async def find_reusable_document(
    db: AsyncSession, user: User, content_hash: str
) -> Document | None:
    """查找该用户已成功处理过的、内容完全相同的文档。

    只复用 completed 的：之前处理失败（例如 Embedding API 出错）的视为没上传过，需要重新处理。
    只在同一用户内查找：不同用户的文档相互隔离，不能把别人的文档返回给你。
    """
    stmt = (
        select(Document)
        .where(
            Document.user_id == user.id,
            Document.content_hash == content_hash,
            Document.status == "completed",
        )
        .order_by(Document.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def upload_document(
    db: AsyncSession, user: User, file: UploadFile
) -> tuple[Document, bool]:
    """上传并处理文档。返回 (文档, 是否为重复上传)。"""
    settings = get_settings()

    # 1. 校验文件类型
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_TYPES:
        raise ValidationError(f"不支持的文件类型：{ext}，支持{', '.join(sorted(ALLOWED_TYPES))}")

    # 2. 校验文件大小
    content = await file.read()
    file_size = len(content)
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise ValidationError(f"文件大小{file_size // 1024 // 1024}MB 超过限制{settings.max_file_size_mb}MB")

    # 2.5 内容去重：按文件字节算哈希（与文件名无关），同一用户已处理过就直接返回
    content_hash = hashlib.sha256(content).hexdigest()
    existing = await find_reusable_document(db, user, content_hash)
    if existing:
        logger.info(
            "document.duplicate_upload",
            document_id=existing.id,
            existing_filename=existing.filename,
            uploaded_filename=filename,
            user_id=user.id,
        )
        return existing, True

    # 3. 创建 Document 记录
    doc = Document(
        user_id=user.id,
        filename=filename,
        file_type=ext,
        file_size=file_size,
        content_hash=content_hash,
        status="pending",
    )
    db.add(doc)
    await db.flush()

    # 4. 保存临时文件 + 处理
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        await process_document(db, doc, temp_path)

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

    # 处理失败时 process_document 不抛异常：文档以 failed 状态正常提交，用户能看到失败原因
    if doc.status == "failed":
        logger.warning(
            "document.processing_failed",
            document_id=doc.id,
            filename=filename,
            error=doc.error_message,
        )
    else:
        logger.info(
            "document.uploaded",
            document_id=doc.id,
            filename=filename,
            file_type=ext,
            chunk_count=doc.chunk_count,
        )

    return doc, False


async def list_documents(
    db: AsyncSession, user: User, skip: int = 0, limit: int = 20
) -> tuple[list[Document], int]:
    count_stmt = select(func.count()).select_from(Document).where(
        Document.user_id == user.id
    )
    total = (await db.execute(count_stmt)).scalar()

    stmt = (
        select(Document)
        .where(Document.user_id == user.id)
        .order_by(Document.created_at.desc())
        .offset(skip).limit(limit)
    )
    result = await db.execute(stmt)
    items = list(result.scalars().all())
    return items, total


async def get_document(db: AsyncSession, user: User, doc_id: int) -> Document:
    stmt = select(Document).where(Document.id == doc_id, Document.user_id == user.id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError("文档", doc_id)
    return doc


async def get_document_chunks(
    db: AsyncSession, user: User, doc_id: int
) -> tuple[Document, list[Chunk]]:
    doc = await get_document(db, user, doc_id)
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == doc.id)
        .order_by(Chunk.chunk_index)
    )
    result = await db.execute(stmt)
    chunks = list(result.scalars().all())
    return doc, chunks


async def delete_document(db: AsyncSession, user: User, doc_id: int) -> None:
    doc = await get_document(db, user, doc_id)
    await db.delete(doc)  # cascade 会自动删除 chunks
    await db.flush()
    logger.info("document.deleted", document_id=doc.id)