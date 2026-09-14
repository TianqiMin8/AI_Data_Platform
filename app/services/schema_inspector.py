from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.datasource import (
    SchemaIntrospectionResponse,
    TableSchema,
    ColumnSchema,
)
from app.services.datasource import get_datasource, _build_sync_url
from app.core.security import decrypt_value
from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def introspect(
    db: AsyncSession, user: User, datasource_id: int
) -> SchemaIntrospectionResponse:
    ds = await get_datasource(db, user, datasource_id)
    plain_password = decrypt_value(ds.encrypted_password)
    url = _build_sync_url(ds, plain_password)

    try:
        engine = create_engine(url, connect_args={"connect_timeout": 10})
        inspector = inspect(engine)

        tables = []
        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append(
                    ColumnSchema(
                        column_name=col["name"],
                        column_type=str(col["type"]),
                        is_nullable=col.get("nullable", True),
                        comment=col.get("comment"),
                    )
                )
            tables.append(
                TableSchema(table_name=table_name, columns=columns)
            )

        engine.dispose()

        logger.info(
            "schema.introspected",
            datasource_id=ds.id,
            table_count=len(tables),
        )

        return SchemaIntrospectionResponse(
            datasource_id=ds.id,
            datasource_name=ds.name,
            tables=tables,
            table_count=len(tables),
        )

    except Exception as e:
        logger.error("schema.introspection_failed", datasource_id=ds.id, error=str(e))
        raise AppError(
            code="SCHEMA_INTROSPECTION_ERROR",
            message=f"Schema 读取失败:{str(e)}",
            status_code=502,
        )