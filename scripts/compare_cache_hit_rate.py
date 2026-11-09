"""作业 5-1：对比精确匹配和语义匹配的缓存命中率。

用真实的 text-embedding-3-small 生成向量（会调用 OpenAI，约 20 条短文本，费用可忽略），
不读写 Redis，在内存里模拟缓存。

用法：
    poetry run python scripts/compare_cache_hit_rate.py
"""

import asyncio

from app.rag.embedder import get_embedder
from app.sql_engine.semantic_cache import (
    CachedEntry,
    cosine_similarity,
    match_cached_entry,
)

THRESHOLD = 0.95

# 已经缓存过的问题
CACHED_QUESTIONS = [
    "总销售额是多少？",
    "每个品类的销售额是多少？",
    "今天的销售额是多少？",
    "销售额最高的前3个客户是谁？",
]

# (新问题, 应该命中的缓存问题；None 表示不应该命中)
QUERIES = [
    ("总销售额是多少？", "总销售额是多少？"),
    ("总销售额是多少", "总销售额是多少？"),
    ("销售总额是多少？", "总销售额是多少？"),
    ("一共卖了多少钱？", "总销售额是多少？"),
    ("总共的销售额是多少", "总销售额是多少？"),
    ("各个品类的销售额分别是多少？", "每个品类的销售额是多少？"),
    ("每个类别卖了多少钱？", "每个品类的销售额是多少？"),
    ("按品类统计销售额", "每个品类的销售额是多少？"),
    ("今天卖了多少钱？", "今天的销售额是多少？"),
    ("销售额最高的前3个客户是哪些？", "销售额最高的前3个客户是谁？"),
    ("昨天的销售额是多少？", None),
    ("本月的销售额是多少？", None),
    ("销售额最高的前5个客户是谁？", None),
    ("平均订单金额是多少？", None),
]


def _exact_match(question: str) -> str | None:
    # 和 cache.py 的精确缓存同样的规范化：去首尾空格 + 小写
    normalized = question.strip().lower()
    for cached in CACHED_QUESTIONS:
        if cached.strip().lower() == normalized:
            return cached
    return None


def _semantic_no_guard(question_vec, entries, threshold) -> CachedEntry | None:
    """不做时间/数字条件检查的纯语义匹配，用来对比条件保护的作用。"""
    best = max(entries, key=lambda e: cosine_similarity(question_vec, e.embedding))
    return best if cosine_similarity(question_vec, best.embedding) > threshold else None


def _score(results, label):
    should_hit = [(exp, got) for exp, got in results if exp is not None]
    should_miss = [(exp, got) for exp, got in results if exp is None]
    correct_hits = sum(1 for exp, got in should_hit if got == exp)
    false_hits = sum(1 for exp, got in should_hit if got is not None and got != exp)
    false_hits += sum(1 for _, got in should_miss if got is not None)
    print(
        f"{label:<26} 命中率 {correct_hits}/{len(should_hit)} = {correct_hits / len(should_hit):.0%}"
        f"    误命中 {false_hits} 次"
    )


async def main():
    embedder = get_embedder()
    cached_vecs = await embedder.embed_batch(CACHED_QUESTIONS)
    query_vecs = await embedder.embed_batch([q for q, _ in QUERIES])

    entries = [
        CachedEntry(question=q, embedding=v, sql=f"<SQL of {q}>")
        for q, v in zip(CACHED_QUESTIONS, cached_vecs)
    ]

    print(f"\n{'新问题':<24}{'最相似的缓存问题':<26}{'相似度':>8}  {'精确':<4}{'语义':<4}{'应命中':<4}")
    exact_results, semantic_results, no_guard_results = [], [], []
    for (question, expected), vec in zip(QUERIES, query_vecs):
        best = max(entries, key=lambda e: cosine_similarity(vec, e.embedding))
        best_sim = cosine_similarity(vec, best.embedding)

        exact = _exact_match(question)
        match = match_cached_entry(question, vec, entries, THRESHOLD)
        semantic = match.entry.question if match.entry else None
        no_guard = _semantic_no_guard(vec, entries, THRESHOLD)

        exact_results.append((expected, exact))
        semantic_results.append((expected, semantic))
        no_guard_results.append((expected, no_guard.question if no_guard else None))

        mark = lambda got: "✓" if got else "·"
        note = "（条件不一致被拦）" if match.blocked_by_constraints else ""
        print(
            f"{question:<22}{best.question:<22}{best_sim:>8.4f}  "
            f"{mark(exact):<4}{mark(semantic):<4}{'是' if expected else '否':<4}{note}"
        )

    print(f"\n阈值 {THRESHOLD}：")
    _score(exact_results, "精确匹配")
    _score(no_guard_results, "语义匹配（无条件保护）")
    _score(semantic_results, "语义匹配（有条件保护）")

    print("\n不同阈值下的语义匹配：")
    for threshold in (0.80, 0.85, 0.90, 0.95):
        guarded, unguarded = [], []
        for (question, expected), vec in zip(QUERIES, query_vecs):
            match = match_cached_entry(question, vec, entries, threshold)
            guarded.append((expected, match.entry.question if match.entry else None))
            no_guard = _semantic_no_guard(vec, entries, threshold)
            unguarded.append((expected, no_guard.question if no_guard else None))
        _score(unguarded, f"  阈值 {threshold} 无条件保护")
        _score(guarded, f"  阈值 {threshold} 有条件保护")


if __name__ == "__main__":
    asyncio.run(main())
