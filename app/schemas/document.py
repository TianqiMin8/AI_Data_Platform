from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    error_message: Optional[str] = None
    chunk_count: int
    total_characters: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class ChunkResponse(BaseModel):
    id: int
    chunk_index: int
    content: str
    char_count: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    source: Optional[str] = None

    model_config = {"from_attributes": True}


class DocumentChunksResponse(BaseModel):
    document_id: int
    filename: str
    chunks: list[ChunkResponse]
    total: int