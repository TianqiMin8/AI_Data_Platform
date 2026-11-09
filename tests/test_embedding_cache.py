"""作业 7-1：Embedding 缓存。用假 Redis 和假 OpenAI 客户端，统计哪些文本真的调用了 API。"""

import asyncio
import hashlib
import math
from types import SimpleNamespace

import pytest

import app.rag.embedder as embedder_module
from app.rag.embedder import (
    Embedder,
    deserialize_vector,
    embedding_cache_key,
    serialize_vector,
)


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    async def mget(self, keys):
        return [self.data.get(k) for k in keys]

    def pipeline(self):
        redis = self

        class Pipeline:
            def __init__(self):
                self.ops = []

            def set(self, key, value, ex=None):
                self.ops.append((key, value, ex))
                return self

            async def execute(self):
                for key, value, ex in self.ops:
                    redis.data[key] = value
                    redis.ttls[key] = ex

        return Pipeline()


class FakeEmbeddingsAPI:
    """每段文本生成一个确定的向量，并记录被请求过的文本。"""

    def __init__(self):
        self.requested: list[str] = []

    async def create(self, model, input):
        self.requested.extend(input)
        data = [SimpleNamespace(embedding=[float(len(t)), 0.5, -0.25]) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=len(input)))


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(embedder_module, "get_redis_client", lambda: fake)
    return fake


def _embedder(model: str = "text-embedding-3-small") -> tuple[Embedder, FakeEmbeddingsAPI]:
    embedder = Embedder()
    api = FakeEmbeddingsAPI()
    embedder.client = SimpleNamespace(embeddings=api)
    embedder.model = model
    return embedder, api


def _embed(embedder, texts):
    return asyncio.run(embedder.embed_batch(texts))


# ── key 与序列化 ──


def test_cache_key_format_includes_model():
    digest = hashlib.sha256("售后政策".encode()).hexdigest()[:16]
    assert embedding_cache_key("售后政策", "text-embedding-3-small") == f"embedding_cache:text-embedding-3-small:{digest}"


def test_serialization_roundtrip_keeps_float32_precision():
    vector = [0.123456789, -0.987654321, 0.0, 1e-7]
    restored = deserialize_vector(serialize_vector(vector))
    assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) for a, b in zip(vector, restored))


def test_serialized_vector_is_compact():
    vector = [0.0123456789] * 1536
    # float32 + base64 约 8KB；同样的数据存成 JSON 文本大约要 20KB 以上
    assert len(serialize_vector(vector)) == 8192


# ── 缓存行为 ──


def test_second_call_hits_cache(redis):
    embedder, api = _embedder()
    first = _embed(embedder, ["销售数据", "售后政策"])
    second = _embed(embedder, ["销售数据", "售后政策"])
    assert api.requested == ["销售数据", "售后政策"]  # 第二次没有调用 API
    assert second == first


def test_only_missing_texts_call_api_and_order_is_kept(redis):
    embedder, api = _embedder()
    _embed(embedder, ["A段", "B段落"])
    api.requested.clear()

    result = _embed(embedder, ["新增的C段落内容", "A段", "B段落"])
    assert api.requested == ["新增的C段落内容"]
    assert [v[0] for v in result] == [len("新增的C段落内容"), len("A段"), len("B段落")]


def test_duplicate_texts_in_one_call_are_requested_once(redis):
    embedder, api = _embedder()
    result = _embed(embedder, ["免责声明", "正文", "免责声明"])
    assert api.requested == ["免责声明", "正文"]
    assert result[0] == result[2]


def test_different_model_does_not_reuse_cache(redis):
    small, small_api = _embedder("text-embedding-3-small")
    large, large_api = _embedder("text-embedding-3-large")
    _embed(small, ["售后政策"])
    _embed(large, ["售后政策"])
    assert small_api.requested == ["售后政策"]
    assert large_api.requested == ["售后政策"]  # 换模型后重新生成，没有读到 small 的向量


def test_ttl_comes_from_settings(redis, monkeypatch):
    monkeypatch.setattr(embedder_module.get_settings(), "embedding_cache_ttl_seconds", 123)
    embedder, _ = _embedder()
    _embed(embedder, ["售后政策"])
    assert list(redis.ttls.values()) == [123]


def test_redis_down_falls_back_to_api(monkeypatch):
    def broken():
        raise ConnectionError("redis down")
    monkeypatch.setattr(embedder_module, "get_redis_client", broken)

    embedder, api = _embedder()
    result = _embed(embedder, ["销售数据"])
    assert api.requested == ["销售数据"]
    assert len(result) == 1


def test_cache_can_be_disabled(redis, monkeypatch):
    monkeypatch.setattr(embedder_module.get_settings(), "embedding_cache_enabled", False)
    embedder, api = _embedder()
    _embed(embedder, ["销售数据"])
    _embed(embedder, ["销售数据"])
    assert api.requested == ["销售数据", "销售数据"]
    assert redis.data == {}
