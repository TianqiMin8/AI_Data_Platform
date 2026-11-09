import redis.asyncio as aioredis
from app.config.settings import get_settings


def get_redis_client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
    )