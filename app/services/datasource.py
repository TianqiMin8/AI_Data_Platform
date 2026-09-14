from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import create_engine

from app.models.datasource import DataSource
from app.models.user import User
from app.schemas.datasource import DataSourceCreate, DataSourceUpdate
from app.core.security import encrypt_value, decrypt_value
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger

from sqlalchemy import select, func, text

logger = get_logger(__name__)


def _build_sync_url(ds: DataSource, plain_password: str) -> str:
    """构造同步连接 URL（用于测试连接和 Schema 自省）。"""
    return _build_sync_url_from_values(
        db_type=ds.db_type,
        host=ds.host,
        port=ds.port,
        database_name=ds.database_name,
        username=ds.username,
        password=plain_password,
    )

def _build_sync_url_from_values(
    db_type: str,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password: str,
) -> str:
    driver = {
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
    }

    prefix = driver.get(db_type, db_type)

    return (
        f"{prefix}://"
        f"{username}:{password}@"
        f"{host}:{port}/"
        f"{database_name}"
    )

async def create_datasource(
    db: AsyncSession, user: User, req: DataSourceCreate
) -> DataSource:
    ds = DataSource(
        user_id=user.id,
        name=req.name,
        db_type=req.db_type,
        host=req.host,
        port=req.port,
        database_name=req.database_name,
        username=req.username,
        encrypted_password=encrypt_value(req.password),
    )
    db.add(ds)
    await db.flush()

    logger.info("datasource.created", datasource_id=ds.id, user_id=user.id, db_type=ds.db_type)
    return ds


async def list_datasources(
    db: AsyncSession,
    user: User,
    cursor: int | None = None,
    limit: int = 20,
) -> tuple[list[DataSource], int | None]:

    stmt = (
        select(DataSource)
        .where(
            DataSource.user_id == user.id,
            DataSource.is_active == True,
        )
        .order_by(DataSource.id.desc())
    )

    if cursor is not None:
        stmt = stmt.where(DataSource.id < cursor)

    # 多取一条，用来判断是否还有下一页
    stmt = stmt.limit(limit + 1)

    result = await db.execute(stmt)
    items = list(result.scalars().all())

    next_cursor = None

    if len(items) > limit:
        items = items[:limit]
        next_cursor = items[-1].id

    return items, next_cursor


async def get_datasource(db: AsyncSession, user: User, datasource_id: int) -> DataSource:
    stmt = select(DataSource).where(
        DataSource.id == datasource_id,
        DataSource.user_id == user.id,
        DataSource.is_active == True,
    )
    result = await db.execute(stmt)
    ds = result.scalar_one_or_none()
    if not ds:
        raise NotFoundError("数据源", datasource_id)
    return ds


async def delete_datasource(db: AsyncSession, user: User, datasource_id: int) -> None:
    ds = await get_datasource(db, user, datasource_id)
    ds.is_active = False
    await db.flush()
    logger.info("datasource.deleted", datasource_id=ds.id, user_id=user.id)


async def update_datasource(
    db: AsyncSession,
    user: User,
    datasource_id: int,
    req: DataSourceUpdate,
) -> DataSource:

    # 1. 找到当前数据源
    ds = await get_datasource(db, user, datasource_id)

    # 2. 解密当前密码
    old_password = decrypt_value(ds.encrypted_password)

    # 3. 构造更新后的候选值
    new_name = req.name if req.name is not None else ds.name
    new_host = req.host if req.host is not None else ds.host
    new_port = req.port if req.port is not None else ds.port
    new_database_name = (
        req.database_name
        if req.database_name is not None
        else ds.database_name
    )
    new_username = (
        req.username
        if req.username is not None
        else ds.username
    )

    # password 比较特殊：
    # 不传 password = 保留旧密码
    if "password" in req.model_fields_set:
        new_password = req.password
    else:
        new_password = old_password

    # 4. 判断连接配置是否真的改变
    connection_changed = (
        new_host != ds.host
        or new_port != ds.port
        or new_database_name != ds.database_name
        or new_username != ds.username
        or new_password != old_password
    )

    # 5. 如果连接配置改变，先测试新连接
    if connection_changed:
        url = _build_sync_url_from_values(
            db_type=ds.db_type,
            host=new_host,
            port=new_port,
            database_name=new_database_name,
            username=new_username,
            password=new_password,
        )

        success, message = _test_connection_url(url)

        if not success:
            raise ValidationError(message)

    # 6. 测试通过后才真正修改 ORM 对象
    ds.name = new_name
    ds.host = new_host
    ds.port = new_port
    ds.database_name = new_database_name
    ds.username = new_username

    # 只有用户明确传了 password 才重新加密
    if "password" in req.model_fields_set:
        ds.encrypted_password = encrypt_value(new_password)

    # 7. 写入数据库
    await db.flush()

    logger.info(
        "datasource.updated",
        datasource_id=ds.id,
        user_id=user.id,
        connection_changed=connection_changed,
    )

    return ds

async def test_connection(
    db: AsyncSession,
    user: User,
    datasource_id: int,
) -> tuple[bool, str]:

    ds = await get_datasource(db, user, datasource_id)

    plain_password = decrypt_value(ds.encrypted_password)

    url = _build_sync_url(ds, plain_password)

    success, message = _test_connection_url(url)

    if not success:
        logger.warning(
            "datasource.connection_failed",
            datasource_id=ds.id,
            user_id=user.id,
        )

    return success, message

def _test_connection_url(url: str) -> tuple[bool, str]: 
    """测试给定数据库 URL 是否可以连接。""" 
    engine = None 
 
    try: 
        engine = create_engine( 
            url, 
            connect_args={"connect_timeout": 5}, 
        ) 
 
        with engine.connect() as conn: 
            conn.execute(text("SELECT 1")) 
 
        return True, "连接成功" 
 
    except Exception as e: 
        logger.warning( 
            "datasource.connection_failed", 
            error=str(e), 
        ) 
        return False, f"连接失败: {str(e)}" 
 
    finally: 
        if engine is not None: 
            engine.dispose()