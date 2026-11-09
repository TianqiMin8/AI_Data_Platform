from fastapi import APIRouter, Depends, UploadFile, File, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.document import (
    DocumentResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentChunksResponse,
    ChunkResponse,
)
from app.services import document as doc_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc, is_duplicate = await doc_service.upload_document(db, current_user, file)
    if is_duplicate:
        response.status_code = 200  # 没有创建新资源，返回的是已有文档
    return DocumentUploadResponse.model_validate(doc).model_copy(update={"is_duplicate": is_duplicate})


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items, total = await doc_service.list_documents(db, current_user, skip, limit)
    return DocumentListResponse(items=items, total=total)


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc, chunks = await doc_service.get_document_chunks(db, current_user, document_id)
    return DocumentChunksResponse(
        document_id=doc.id,
        filename=doc.filename,
        chunks=chunks,
        total=len(chunks),
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await doc_service.delete_document(db, current_user, document_id)