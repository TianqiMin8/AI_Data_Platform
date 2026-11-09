from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, Chunk
from app.rag.parser import get_parser, ParsedDocument
from app.rag.chunker import get_chunker, ChunkResult
from app.config.settings import get_settings
from app.core.logging import get_logger
from app.rag.embedder import get_embedder

logger = get_logger(__name__)


async def process_document(
    db: AsyncSession,
    document: Document,
    file_path: str,
) -> Document:
    """完整的文档处理 Pipeline：解析 → 切片 → 入库。"""

    settings = get_settings()

    try:
        # 1. 更新状态
        document.status = "processing"
        await db.flush()

        # 2. 解析
        parser = get_parser(document.file_type)
        parsed: ParsedDocument = parser.parse(file_path)

        if not parsed.pages:
            raise ValueError("文档解析后无内容")

        logger.info(
            "pipeline.parsed",
            document_id=document.id,
            pages=len(parsed.pages),
            total_characters=parsed.total_characters,
        )

        # 3. 切片
        chunker = get_chunker(
            strategy=settings.chunk_strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        all_chunks: list[ChunkResult] = []
        for page in parsed.pages:
            if not page.content.strip():
                continue
            page_chunks = chunker.chunk(page.content)
            # 把页面级 metadata 传递给每个 chunk
            for c in page_chunks:
                c.page_number = c.page_number or page.page_number
                c.section_title = c.section_title or page.section_title
            all_chunks.extend(page_chunks)

        # 重新编号 chunk_index（跨页连续）
        for i, c in enumerate(all_chunks):
            c.chunk_index = i

        if not all_chunks:
            raise ValueError("切片后无有效内容")

        logger.info(
            "pipeline.chunked",
            document_id=document.id,
            chunk_count=len(all_chunks),
            strategy=settings.chunk_strategy,
        )

        # 4. 批量入库
        chunk_records = []
        for c in all_chunks:
            chunk_records.append(Chunk(
                document_id=document.id,
                chunk_index=c.chunk_index,
                content=c.content,
                char_count=c.char_count,
                page_number=c.page_number,
                section_title=c.section_title,
                source=document.filename,
            ))

        db.add_all(chunk_records)
        await db.flush()

         # ✅ 6. 生成 Embedding
        document.status = "embedding"
        await db.flush()

        embedder = get_embedder()
        texts = [c.content for c in chunk_records]

        try:
            embeddings = await embedder.embed_batch(texts)

            for chunk_record, embedding in zip(chunk_records, embeddings):
                chunk_record.embedding = embedding
                chunk_record.embedding_model = embedder.model

            await db.flush()

            logger.info(
                "pipeline.embedding_completed",
                document_id=document.id,
                chunk_count=len(embeddings),
            )

        except Exception as e:
            logger.error("pipeline.embedding_failed", document_id=document.id, error=str(e))
            document.status = "failed"
            document.error_message = f"Embedding 生成失败：{str(e)}"
            await db.flush()
            raise

        # 5. 更新 Document 状态
        document.status = "completed"
        document.chunk_count = len(chunk_records)
        document.total_characters = sum(c.char_count for c in all_chunks)
        await db.flush()

        logger.info(
            "pipeline.completed",
            document_id=document.id,
            chunk_count=document.chunk_count,
        )

        return document

    except Exception as e:
        document.status = "failed"
        document.error_message = str(e)[:500]
        await db.flush()

        logger.error(
            "pipeline.failed",
            document_id=document.id,
            error=str(e),
        )
        raise