from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.rag.retriever import vector_search, bm25_search, hybrid_search
from app.rag.reranker import rerank
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from app.core.logging import get_logger

logger = get_logger(__name__)


async def search_documents(
    db: AsyncSession, user: User, req: SearchRequest
) -> SearchResponse:
    """统一的文档检索入口。"""

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
        results = await rerank(req.query, results, top_k=req.top_k)
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
    )

    return SearchResponse(
        query=req.query,
        results=items,
        total=len(items),
        search_method=req.search_method,
    )