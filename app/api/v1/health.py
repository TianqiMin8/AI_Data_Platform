from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.config.settings import get_settings
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

# 目前只检查是否拿到settings和数据库连接是否正常，后续可以增加更多检查项，比如缓存、消息队列等
@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/health/db")
async def health_check_db(db: AsyncSession = Depends(get_db)):
    # When database has connection error, return real error message
    try:
        result = await db.execute(text("SELECT 1"))
        row = result.scalar()
        return {"status": "ok", "db": row == 1}
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "db": False,
                "error": str(e),
            },
        )
