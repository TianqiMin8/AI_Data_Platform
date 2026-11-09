from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.stats import CacheStatsResponse
from app.sql_engine.cache_stats import get_cache_stats_and_warn

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/cache", response_model=CacheStatsResponse)
async def cache_stats(current_user: User = Depends(get_current_user)):
    return await get_cache_stats_and_warn()
