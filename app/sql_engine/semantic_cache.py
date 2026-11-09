"""语义相似度缓存：问题换个说法也能命中缓存的 SQL。

存储：每个数据源一个 Redis Hash，key = sem_cache:{datasource_id}
      field = 问题的哈希，value = JSON {question, embedding, sql}
查找：新问题生成 Embedding，和该数据源所有缓存问题逐条算余弦相似度，
      超过阈值且"时间/数字条件"一致才命中。
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass

from app.config.settings import get_settings
from app.core.logging import get_logger
from app.db.redis import get_redis_client
from app.rag.embedder import get_embedder

logger = get_logger(__name__)


@dataclass
class CachedEntry:
    question: str
    embedding: list[float]
    sql: str


@dataclass
class SemanticMatch:
    entry: CachedEntry | None      # 命中的缓存条目；未命中为 None
    similarity: float              # 命中条目的相似度；未命中时为最高相似度
    blocked_by_constraints: bool   # 有条目超过阈值，但时间/数字条件不一致被拦下


def _semantic_key(datasource_id: int) -> str:
    return f"sem_cache:{datasource_id}"


def _question_field(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


# ── 相似度计算 ──


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── 时间 / 数字条件 ──
# "今天的销售额" 和 "昨天的销售额" 向量很接近，但 SQL 的 WHERE 条件不同，不能复用。
# 提取问题里的时间词和数字，两个问题的条件集合必须完全一致才允许命中。

TIME_WORDS = [
    "今天", "昨天", "前天", "明天",
    "本周", "这周", "上周", "下周",
    "本月", "这个月", "上个月", "上月", "下个月",
    "本季度", "上季度", "今年", "去年", "前年", "明年",
    "最近", "近期", "今日", "昨日",
]

CONSTRAINT_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?"),               # 2026、3、10.5
    re.compile(r"第[一二三四五六七八九十]+"),     # 第一季度、第三
    re.compile(r"前[一二三四五六七八九十百]+"),   # 前十名、前三
]


def extract_constraints(question: str) -> frozenset[str]:
    """提取问题中会影响 SQL 条件的时间词和数字。"""
    found = {word for word in TIME_WORDS if word in question}
    for pattern in CONSTRAINT_PATTERNS:
        found.update(pattern.findall(question))
    return frozenset(found)


def match_cached_entry(
    question: str,
    embedding: list[float],
    entries: list[CachedEntry],
    threshold: float,
) -> SemanticMatch:
    """在缓存条目中找相似度最高、且条件一致的一条。纯函数，方便测试和离线对比。"""
    question_constraints = extract_constraints(question)

    best_similarity = 0.0
    best_entry: CachedEntry | None = None
    blocked = False

    scored = sorted(
        ((cosine_similarity(embedding, e.embedding), e) for e in entries),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for similarity, entry in scored:
        best_similarity = max(best_similarity, similarity)
        if similarity <= threshold:
            break  # 已按相似度降序，后面的更低
        if extract_constraints(entry.question) != question_constraints:
            blocked = True
            continue
        best_entry = entry
        best_similarity = similarity
        break

    return SemanticMatch(entry=best_entry, similarity=best_similarity, blocked_by_constraints=blocked)


# ── Redis 读写 ──


async def find_similar_cached_sql(
    datasource_id: int, question: str
) -> tuple[SemanticMatch | None, list[float] | None]:
    """查找语义相似的缓存 SQL。

    返回 (匹配结果, 问题的 Embedding)。Embedding 一并返回，后面写缓存时复用，避免重复调用 API。
    Embedding 或 Redis 出错时返回 (None, ...)，不阻断主链路。
    """
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return None, None

    try:
        embedding = await get_embedder().embed_query(question)
    except Exception as e:
        logger.warning("sql.semantic_cache.embed_failed", error=str(e))
        return None, None

    try:
        client = get_redis_client()
        raw_entries = await client.hgetall(_semantic_key(datasource_id))
    except Exception as e:
        logger.warning("sql.semantic_cache.get_failed", error=str(e))
        return None, embedding

    entries = [CachedEntry(**json.loads(raw)) for raw in raw_entries.values()]
    match = match_cached_entry(question, embedding, entries, settings.semantic_cache_threshold)

    if match.entry:
        logger.info(
            "sql.semantic_cache.hit",
            question=question[:50],
            matched_question=match.entry.question[:50],
            similarity=round(match.similarity, 4),
        )
    elif match.blocked_by_constraints:
        logger.info(
            "sql.semantic_cache.constraint_mismatch",
            question=question[:50],
            similarity=round(match.similarity, 4),
        )
    else:
        logger.info(
            "sql.semantic_cache.miss",
            question=question[:50],
            best_similarity=round(match.similarity, 4),
            cached_count=len(entries),
        )
    return match, embedding


async def set_semantic_cached_sql(
    datasource_id: int,
    question: str,
    sql: str,
    embedding: list[float] | None = None,
) -> None:
    """写入语义缓存。embedding 为空时现算一次。"""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return

    try:
        if embedding is None:
            embedding = await get_embedder().embed_query(question)

        entry = CachedEntry(question=question, embedding=embedding, sql=sql)
        key = _semantic_key(datasource_id)
        client = get_redis_client()
        await client.hset(key, _question_field(question), json.dumps(entry.__dict__))
        await client.expire(key, settings.cache_ttl_seconds)
        logger.info("sql.semantic_cache.set", key=key, question=question[:50])
    except Exception as e:
        logger.warning("sql.semantic_cache.set_failed", error=str(e))
