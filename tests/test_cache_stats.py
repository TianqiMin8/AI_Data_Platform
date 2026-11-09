"""作业 5-4：缓存命中率监控。用内存里的假 Redis，不影响真实计数。"""

import asyncio

import pytest

import app.sql_engine.cache_stats as cache_stats
from app.sql_engine.cache_stats import get_cache_stats, get_cache_stats_and_warn, record_cache_result


class FakeRedis:
    def __init__(self):
        self.data: dict[str, int] = {}

    async def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    async def mget(self, *keys):
        return [str(self.data[k]) if k in self.data else None for k in keys]


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache_stats, "get_redis_client", lambda: redis)
    return redis


@pytest.fixture
def warnings(monkeypatch):
    """收集 cache.hit_rate_low 告警。"""
    captured = []
    original = cache_stats.logger.warning

    def capture(event, **kwargs):
        if event == "cache.hit_rate_low":
            captured.append(kwargs)
        return original(event, **kwargs)

    monkeypatch.setattr(cache_stats.logger, "warning", capture)
    return captured


def _record(results: list):
    async def run():
        for r in results:
            await record_cache_result(r)
    asyncio.run(run())


def test_counts_hits_and_misses_by_source(fake_redis):
    _record(["exact", "exact", "semantic", None, None])
    stats = asyncio.run(get_cache_stats())
    assert stats["hit_count"] == 3
    assert stats["miss_count"] == 2
    assert stats["total"] == 5
    assert stats["hit_rate"] == 0.6
    assert stats["exact_hit_count"] == 2
    assert stats["semantic_hit_count"] == 1


def test_empty_stats(fake_redis):
    stats = asyncio.run(get_cache_stats())
    assert stats["total"] == 0
    assert stats["hit_rate"] == 0.0
    assert not stats["below_threshold"]


def test_low_hit_rate_warns_after_enough_samples(fake_redis, warnings):
    # 20 次查找只命中 2 次 = 10% < 20%
    _record(["exact", "semantic"] + [None] * 18)
    assert len(warnings) == 1
    assert warnings[0]["hit_rate"] == 0.1


def test_few_samples_do_not_warn(fake_redis, warnings):
    # 刚启动时 0/5 = 0%，但样本太少，不告警
    _record([None] * 5)
    assert warnings == []
    assert not asyncio.run(get_cache_stats())["below_threshold"]


def test_healthy_hit_rate_does_not_warn(fake_redis, warnings):
    _record(["exact"] * 10 + [None] * 10)  # 50%
    assert warnings == []


def test_warning_is_throttled(fake_redis, warnings):
    # 只在每累计 20 次查找时检查一次，不会每次请求都刷屏
    _record([None] * 39)
    assert len(warnings) == 1
    _record([None])
    assert len(warnings) == 2


def test_api_helper_warns_when_low(fake_redis, warnings):
    _record([None] * 25)
    warnings.clear()
    stats = asyncio.run(get_cache_stats_and_warn())
    assert stats["below_threshold"]
    assert len(warnings) == 1


def test_redis_failure_does_not_raise(monkeypatch):
    def broken():
        raise ConnectionError("redis down")
    monkeypatch.setattr(cache_stats, "get_redis_client", broken)
    asyncio.run(record_cache_result("exact"))  # 不抛异常即通过
