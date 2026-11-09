import asyncio
from openai import AsyncOpenAI

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.core.errors import AppError

logger = get_logger(__name__)


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
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.batch_size = 100  # OpenAI 一次最多 2048 条，但 100 条更稳

    async def embed_single(self, text: str) -> list[float]:
        """生成单条文本的 Embedding。"""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 Embedding。"""
        if not texts:
            return []

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