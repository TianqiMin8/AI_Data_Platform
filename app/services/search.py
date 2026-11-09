from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.models.document import Document
from app.models.user import User
from app.rag.retriever import vector_search, bm25_search, hybrid_search
from app.rag.reranker import rerank
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.core.logging import get_logger

logger = get_logger(__name__)


async def _validate_document_ids(db: AsyncSession, user: User, document_ids: list[int]) -> list[int]:
    """确认指定的文档都属于当前用户，返回去重后的 ID。

    检索 SQL 本身已经限定了 d.user_id，查不到别人的数据；这里的作用是：
    ID 写错或不属于自己时明确报错，而不是悄悄返回空结果，让用户分不清是"没搜到"还是"ID 不对"。
    """
    unique_ids = list(dict.fromkeys(document_ids))
    stmt = select(Document.id).where(Document.id.in_(unique_ids), Document.user_id == user.id)
    found = set((await db.execute(stmt)).scalars().all())
    missing = [i for i in unique_ids if i not in found]
    if missing:
        raise NotFoundError("文档", ", ".join(str(i) for i in missing))
    return unique_ids


async def search_documents(
    db: AsyncSession, user: User, req: SearchRequest
) -> SearchResponse:
    """统一的文档检索入口。document_ids 不为空时，只在这些文档的 chunks 中检索。"""

    if req.document_ids:
        req = req.model_copy(update={"document_ids": await _validate_document_ids(db, user, req.document_ids)})

    # 1. 根据方法选择检索策略
    if req.search_method == "vector":
        results = await vector_search(
            db, req.query, user.id, req.top_k * 2, req.document_ids
        )
    elif req.search_method == "bm25":
        results = await bm25_search(
            db, req.query, user.id, req.top_k * 2, req.document_ids
        )
    else:
        results = await hybrid_search(
            db, req.query, user.id, req.top_k * 2, req.document_ids
        )

    # 2. Re-ranking（可选）
    if req.rerank and results:
        results = await rerank(req.query, results, top_k=req.top_k, method=req.rerank_method)
    else:
        results = results[:req.top_k]

    # 3. 转换为响应格式
    items = [
        SearchResultItem(
            chunk_id=r.chunk_id,
            content=r.content,
            score=r.score,
            source=r.source,
            page_number=r.page_number,
            section_title=r.section_title,
            document_id=r.document_id,
            retrieval_method=r.retrieval_method,
        )
        for r in results
    ]

    logger.info(
        "search.completed",
        query=req.query[:50],
        method=req.search_method,
        results=len(items),
        reranked=req.rerank,
        document_ids=req.document_ids,
    )

    return SearchResponse(
        query=req.query,
        results=items,
        total=len(items),
        search_method=req.search_method,
    )