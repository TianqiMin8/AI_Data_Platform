from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, Chunk
from app.rag.parser import get_parser, ParsedDocument
from app.rag.chunker import get_chunker, deduplicate_chunks, ChunkResult
from app.config.settings import get_settings
from app.core.logging import get_logger
from app.rag.embedder import get_embedder, count_embedding_tokens, EMBEDDING_MAX_TOKENS

logger = get_logger(__name__)


async def process_document(
    db: AsyncSession,
    document: Document,
    file_path: str,
) -> Document:
    """完整的文档处理 Pipeline：解析 → 切片 → 入库 → Embedding。

    失败时不抛异常，而是把文档标记为 failed（带上失败阶段和原因）并返回，由调用方正常提交。
    以前失败后会继续抛异常，get_db 随即回滚整个请求，连 Document 记录和失败原因都不会保存。
    """
    document_id = document.id
    progress = {"stage": "processing"}

    try:
        # 处理过程放在 SAVEPOINT 里：任何一步失败都整体撤销（包括已写入的 chunk），
        # 而外层事务里的 Document 记录保留，失败状态才能写进数据库。
        # 数据库本身出错时也一样：SAVEPOINT 回滚后外层事务仍然可用。
        async with db.begin_nested():
            await _run_pipeline(db, document, file_path, progress)
        return document

    except Exception as e:
        stage = progress["stage"]
        logger.error("pipeline.failed", document_id=document_id, stage=stage, error=str(e))

        document.status = "failed"
        document.error_message = f"[{stage}] {e}"[:500]
        document.chunk_count = 0
        await db.flush()
        # SAVEPOINT 回滚会让 document 的属性过期；异步环境下不能懒加载，主动刷新一次
        await db.refresh(document)
        return document


async def _run_pipeline(
    db: AsyncSession,
    document: Document,
    file_path: str,
    progress: dict,
) -> None:
    """Pipeline 的各个步骤。progress["stage"] 记录当前阶段，失败时用来说明卡在哪一步。"""

    settings = get_settings()

    # 1. 更新状态
    document.status = "processing"
    await db.flush()

    # 2. 解析
    progress["stage"] = "parsing"
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
    progress["stage"] = "chunking"
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

    # 3.5 去重：同一文档内内容完全相同的 chunk 只保留第一个
    # 放在重新编号和 Embedding 之前：编号保持连续，重复内容也不浪费 Embedding 调用
    all_chunks, duplicate_count = deduplicate_chunks(all_chunks)
    if duplicate_count:
        logger.info(
            "pipeline.deduplicated",
            document_id=document.id,
            duplicate_count=duplicate_count,
            kept=len(all_chunks),
        )

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

    # 4. 批量入库（同时按 Embedding 模型的分词方式计算 token 数）
    progress["stage"] = "saving"
    chunk_records = []
    for c in all_chunks:
        token_count = count_embedding_tokens(c.content)
        if token_count > EMBEDDING_MAX_TOKENS:
            # 超过 Embedding 模型输入上限，后面调用 API 会失败；先记下是哪一块
            logger.warning(
                "pipeline.chunk_exceeds_embedding_limit",
                document_id=document.id,
                chunk_index=c.chunk_index,
                token_count=token_count,
                limit=EMBEDDING_MAX_TOKENS,
            )
        chunk_records.append(Chunk(
            document_id=document.id,
            chunk_index=c.chunk_index,
            content=c.content,
            char_count=c.char_count,
            token_count=token_count,
            page_number=c.page_number,
            section_title=c.section_title,
            source=document.filename,
        ))

    db.add_all(chunk_records)
    await db.flush()

    # ✅ 6. 生成 Embedding（失败时由 process_document 统一标记为 failed）
    progress["stage"] = "embedding"
    document.status = "embedding"
    await db.flush()

    embedder = get_embedder()
    texts = [c.content for c in chunk_records]
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