from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    document_ids: list[int] | None = None
    search_method: str = Field("hybrid", pattern="^(vector|bm25|hybrid)$")
    rerank: bool = True
    # 不传时使用配置 RERANK_METHOD（默认 cross_encoder）
    rerank_method: str | None = Field(None, pattern="^(llm|cross_encoder)$")


class SearchResultItem(BaseModel):
    chunk_id: int
    content: str
    score: float
    source: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    document_id: Optional[int] = None
    retrieval_method: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total: int
    search_method: str