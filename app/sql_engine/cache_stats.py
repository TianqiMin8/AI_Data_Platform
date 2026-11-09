"""SQL 缓存命中率统计。

Redis 计数器（全局，所有数据源合计）：
    cache_hit_count            命中总数（精确 + 语义）
    cache_miss_count           未命中总数
    cache_hit_count:exact      其中精确匹配命中的次数
    cache_hit_count:semantic   其中语义匹配命中的次数
"""

from typing import Literal

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.db.redis import get_redis_client

logger = get_logger(__name__)

HIT_KEY = "cache_hit_count"
MISS_KEY = "cache_miss_count"
EXACT_HIT_KEY = "cache_hit_count:exact"
SEMANTIC_HIT_KEY = "cache_hit_count:semantic"

CacheSource = Literal["exact", "semantic"]


def _build_stats(hit: int, miss: int, exact_hit: int, semantic_hit: int) -> dict:
    settings = get_settings()
    total = hit + miss
    hit_rate = hit / total if total else 0.0
    return {
        "hit_count": hit,
        "miss_count": miss,
        "total": total,
        "hit_rate": round(hit_rate, 4),
        "exact_hit_count": exact_hit,
        "semantic_hit_count": semantic_hit,
        "warn_threshold": settings.cache_hit_rate_warn_threshold,
        # 样本不足时不判定为过低，避免刚启动时 0/1 就告警
        "below_threshold": total >= settings.cache_hit_rate_min_samples
        and hit_rate < settings.cache_hit_rate_warn_threshold,
    }


def _warn_if_low(stats: dict) -> None:
    if stats["below_threshold"]:
        logger.warning(
            "cache.hit_rate_low",
            hit_rate=stats["hit_rate"],
            hit_count=stats["hit_count"],
            miss_count=stats["miss_count"],
            exact_hit_count=stats["exact_hit_count"],
            semantic_hit_count=stats["semantic_hit_count"],
            threshold=stats["warn_threshold"],
        )


async def record_cache_result(source: CacheSource | None) -> None:
    """记录一次缓存查找的结果：source 为命中的缓存层，None 表示未命中。

    每累计 min_samples 次查找检查一次命中率，过低就输出 warning（不每次都打，避免刷屏）。
    Redis 出错只记日志，不影响主链路。
    """
    try:
        client = get_redis_client()
        if source is None:
            await client.incr(MISS_KEY)
        else:
            await client.incr(HIT_KEY)
            await client.incr(EXACT_HIT_KEY if source == "exact" else SEMANTIC_HIT_KEY)

        stats = await get_cache_stats()
        if stats["total"] % get_settings().cache_hit_rate_min_samples == 0:
            _warn_if_low(stats)
    except Exception as e:
        logger.warning("cache.stats.record_failed", error=str(e))


async def get_cache_stats() -> dict:
    """读取当前命中率统计。"""
    client = get_redis_client()
    values = await client.mget(HIT_KEY, MISS_KEY, EXACT_HIT_KEY, SEMANTIC_HIT_KEY)
    hit, miss, exact_hit, semantic_hit = (int(v or 0) for v in values)
    return _build_stats(hit, miss, exact_hit, semantic_hit)


async def get_cache_stats_and_warn() -> dict:
    """给 API 用：返回统计，命中率过低时同时输出 warning。"""
    stats = await get_cache_stats()
    _warn_if_low(stats)
    return stats
