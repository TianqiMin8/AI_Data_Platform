import hashlib
import json
from app.db.redis import get_redis_client
from app.config.settings import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _cache_key(datasource_id: int, question: str) -> str:
    q_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"sql_cache:{datasource_id}:{q_hash}"


async def get_cached_sql(datasource_id: int, question: str) -> str | None:
    """查找缓存的 SQL。"""
    key = _cache_key(datasource_id, question)
    try:
        client = get_redis_client()
        cached = await client.get(key)
        if cached:
            logger.info("sql.cache.hit", key=key, question=question[:50])
            return cached
        return None
    except Exception as e:
        logger.warning("sql.cache.get_failed", error=str(e))
        return None  # Redis 故障不阻断主链路


async def set_cached_sql(
    datasource_id: int, question: str, sql: str
) -> None:
    """缓存生成的 SQL。"""
    settings = get_settings()
    key = _cache_key(datasource_id, question)
    try:
        client = get_redis_client()
        await client.set(key, sql, ex=settings.cache_ttl_seconds)
        logger.info("sql.cache.set", key=key, ttl=settings.cache_ttl_seconds)
    except Exception as e:
        logger.warning("sql.cache.set_failed", error=str(e))


async def invalidate_cache(datasource_id: int) -> None:
    """清空某个数据源的所有缓存（Schema 变更时调用），包括精确缓存和语义缓存。"""
    try:
        client = get_redis_client()
        pattern = f"sql_cache:{datasource_id}:*"
        keys = [f"sem_cache:{datasource_id}"]
        async for key in client.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await client.delete(*keys)
            logger.info("sql.cache.invalidated", datasource_id=datasource_id, count=len(keys))
    except Exception as e:
        logger.warning("sql.cache.invalidate_failed", error=str(e))