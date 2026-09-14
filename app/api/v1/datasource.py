from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.datasource import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
    DataSourceListResponse,
    ConnectionTestResponse,
    SchemaIntrospectionResponse,
)
from app.services import datasource as ds_service
from app.services import schema_inspector

router = APIRouter(prefix="/datasources", tags=["datasources"])


@router.post("", response_model=DataSourceResponse, status_code=201)
async def create_datasource(
    req: DataSourceCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ds = await ds_service.create_datasource(db, current_user, req)
    return ds


@router.get("", response_model=DataSourceListResponse)
async def list_datasources(
    cursor: int | None = Query(None, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items, next_cursor = await ds_service.list_datasources(
        db,
        current_user,
        cursor,
        limit,
    )

    return DataSourceListResponse(
        items=items,
        next_cursor=next_cursor,
    )

@router.delete("/{datasource_id}", status_code=204)
async def delete_datasource(
    datasource_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ds_service.delete_datasource(db, current_user, datasource_id)


@router.post("/{datasource_id}/test", response_model=ConnectionTestResponse)
async def test_datasource_connection(
    datasource_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success, message = await ds_service.test_connection(db, current_user, datasource_id)
    return ConnectionTestResponse(success=success, message=message)


@router.get("/{datasource_id}/schema", response_model=SchemaIntrospectionResponse)
async def get_datasource_schema(
    datasource_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    
    result = await schema_inspector.introspect(db, current_user, datasource_id)
    return result


@router.put("/{datasource_id}", response_model=DataSourceResponse)
async def update_datasource(
    datasource_id: int,
    req: DataSourceUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ds = await ds_service.update_datasource(
        db,
        current_user,
        datasource_id,
        req,
    )
    return ds