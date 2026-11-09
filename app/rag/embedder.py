import asyncio
import base64
import hashlib
from array import array
from functools import lru_cache

import tiktoken
from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.core.errors import AppError
from app.db.redis import get_redis_client

logger = get_logger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MAX_TOKENS = 8191  # text-embedding-3-small 单条输入的 token 上限


@lru_cache
def _embedding_encoding() -> tiktoken.Encoding:
    # text-embedding-3-small 用 cl100k_base，和 gpt-4o 的 o200k_base 不同，必须按 Embedding 模型取编码器
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_embedding_tokens(text: str) -> int:
    """按 Embedding 模型的分词方式计算 token 数。"""
    return len(_embedding_encoding().encode(text))


# ── Embedding 缓存 ──


def embedding_cache_key(text: str, model: str) -> str:
    """embedding_cache:{模型}:{sha256(内容)[:16]}

    key 里带上模型名：不同模型的向量维度、语义空间都不同，绝对不能混用；
    换模型后自然读不到旧模型的缓存，不需要手动清理。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"embedding_cache:{model}:{digest}"


def serialize_vector(vector: list[float]) -> str:
    """向量按 float32 打包后 base64 编码：1536 维约 8KB，存 JSON 约 30KB。

    OpenAI 的向量本身就是 float32 精度，pgvector 存的也是 float32，这样存不损失精度。
    """
    return base64.b64encode(array("f", vector).tobytes()).decode("ascii")


def deserialize_vector(raw: str) -> list[float]:
    values = array("f")
    values.frombytes(base64.b64decode(raw))
    return values.tolist()


class EmbeddingError(AppError):
    def __init__(self, message: str = "Embedding 生成失败"):
        super().__init__(code="EMBEDDING_ERROR", message=message, status_code=502)


class Embedder:
    """Embedding 生成器，支持批量调用和 rate limit 控制。"""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=30,
        )
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self.batch_size = 100  # OpenAI 一次最多 2048 条，但 100 条更稳

    async def embed_single(self, text: str) -> list[float]:
        """生成单条文本的 Embedding。"""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 Embedding：先查 Redis 缓存，只对没缓存过的文本调用 API，结果写回缓存。

        返回顺序与 texts 一致。Redis 不可用时退化为全部调用 API，不影响主流程。
        """
        if not texts:
            return []

        settings = get_settings()
        if not settings.embedding_cache_enabled:
            return await self._embed_uncached(texts)

        keys = [embedding_cache_key(t, self.model) for t in texts]
        cached = await self._read_cache(keys)

        # 未命中的文本按 key 去重：同一次调用里重复的文本只请求一次
        missing: dict[str, str] = {}
        for key, text, vector in zip(keys, texts, cached):
            if vector is None and key not in missing:
                missing[key] = text

        fresh: dict[str, list[float]] = {}
        if missing:
            vectors = await self._embed_uncached(list(missing.values()))
            fresh = dict(zip(missing.keys(), vectors))
            await self._write_cache(fresh, settings.embedding_cache_ttl_seconds)

        hit_count = sum(v is not None for v in cached)
        logger.info(
            "embedding.cache",
            total=len(texts),
            hits=hit_count,
            api_calls_for=len(missing),
        )
        return [vector if vector is not None else fresh[key] for key, vector in zip(keys, cached)]

    async def _read_cache(self, keys: list[str]) -> list[list[float] | None]:
        try:
            raw_values = await get_redis_client().mget(keys)
            return [deserialize_vector(v) if v else None for v in raw_values]
        except Exception as e:
            logger.warning("embedding.cache.read_failed", error=str(e))
            return [None] * len(keys)

    async def _write_cache(self, vectors: dict[str, list[float]], ttl_seconds: int | None) -> None:
        try:
            pipe = get_redis_client().pipeline()
            for key, vector in vectors.items():
                pipe.set(key, serialize_vector(vector), ex=ttl_seconds)
            await pipe.execute()
        except Exception as e:
            logger.warning("embedding.cache.write_failed", error=str(e))

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """直接调用 Embedding API（按 batch_size 分批）。"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(
                    "embedding.batch.completed",
                    batch_index=i // self.batch_size,
                    batch_size=len(batch),
                    total_tokens=response.usage.total_tokens,
                )

                # Rate limit 保护：批次之间稍微等一下
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(
                    "embedding.batch.failed",
                    batch_index=i // self.batch_size,
                    error=str(e),
                )
                raise EmbeddingError(f"Embedding 生成失败：{str(e)}")

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """生成查询文本的 Embedding（和 embed_single 一样，语义上区分用途）。"""
        return await self.embed_single(query)


def get_embedder() -> Embedder:
    return Embedder()