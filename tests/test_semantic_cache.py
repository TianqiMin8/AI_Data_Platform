"""语义缓存的单元测试：用手工构造的向量，不调用 OpenAI。"""

import math

from app.sql_engine.semantic_cache import (
    CachedEntry,
    cosine_similarity,
    extract_constraints,
    match_cached_entry,
)

THRESHOLD = 0.95


def _vec(angle_deg: float) -> list[float]:
    """二维单位向量。两个向量夹角为 θ 时，余弦相似度 = cos(θ)。"""
    rad = math.radians(angle_deg)
    return [math.cos(rad), math.sin(rad)]


# ── 余弦相似度 ──


def test_cosine_similarity_basics():
    assert math.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0, abs_tol=1e-9)
    assert math.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0  # 零向量不报错


def test_cosine_similarity_ignores_length():
    assert math.isclose(cosine_similarity([1, 1], [5, 5]), 1.0)


# ── 阈值 ──


def test_hit_above_threshold():
    # cos(10°) ≈ 0.985 > 0.95
    entry = CachedEntry("总销售额是多少？", _vec(0), "SELECT SUM(amount) FROM demo_orders")
    match = match_cached_entry("销售总额是多少？", _vec(10), [entry], THRESHOLD)
    assert match.entry is entry
    assert match.similarity > THRESHOLD


def test_miss_below_threshold():
    # cos(30°) ≈ 0.866 < 0.95
    entry = CachedEntry("总销售额是多少？", _vec(0), "SELECT ...")
    match = match_cached_entry("平均订单金额是多少？", _vec(30), [entry], THRESHOLD)
    assert match.entry is None
    assert not match.blocked_by_constraints


def test_picks_most_similar_entry():
    far = CachedEntry("A", _vec(15), "SQL_A")   # cos(15°) ≈ 0.966
    near = CachedEntry("B", _vec(5), "SQL_B")   # cos(5°) ≈ 0.996
    match = match_cached_entry("C", _vec(0), [far, near], THRESHOLD)
    assert match.entry is near


def test_empty_cache_misses():
    match = match_cached_entry("总销售额是多少？", _vec(0), [], THRESHOLD)
    assert match.entry is None
    assert match.similarity == 0.0


# ── 时间 / 数字条件 ──


def test_extract_constraints():
    assert extract_constraints("今天的销售额是多少？") == {"今天"}
    assert extract_constraints("2026年3月的订单有多少笔") == {"2026", "3"}
    assert extract_constraints("销售额前十的客户") == {"前十"}
    assert extract_constraints("第一季度的销售额") == {"第一"}
    assert extract_constraints("总销售额是多少？") == frozenset()


def test_same_vector_different_time_is_blocked():
    # 向量几乎一样，但"今天"≠"昨天"，不能复用 SQL
    entry = CachedEntry("今天的销售额是多少？", _vec(0), "... WHERE created_at >= CURRENT_DATE")
    match = match_cached_entry("昨天的销售额是多少？", _vec(1), [entry], THRESHOLD)
    assert match.entry is None
    assert match.blocked_by_constraints


def test_different_number_is_blocked():
    entry = CachedEntry("销售额最高的前3个客户", _vec(0), "... LIMIT 3")
    match = match_cached_entry("销售额最高的前5个客户", _vec(1), [entry], THRESHOLD)
    assert match.entry is None
    assert match.blocked_by_constraints


def test_falls_through_to_entry_with_matching_constraints():
    # 最相似的条目时间条件不一致，但第二相似的一致且也过阈值 → 命中第二条
    today = CachedEntry("今天的销售额", _vec(0), "SQL_TODAY")
    yesterday = CachedEntry("昨天卖了多少", _vec(8), "SQL_YESTERDAY")
    match = match_cached_entry("昨天的销售额", _vec(1), [today, yesterday], THRESHOLD)
    assert match.entry is yesterday


def test_same_constraints_hit():
    entry = CachedEntry("今天的销售额是多少？", _vec(0), "SQL_TODAY")
    match = match_cached_entry("今天卖了多少钱？", _vec(5), [entry], THRESHOLD)
    assert match.entry is entry
