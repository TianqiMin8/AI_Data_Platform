from app.llm.factory import get_llm_adapter
from app.rag.retriever import RetrievalResult
from app.config.settings import get_settings
from app.core.logging import get_logger
from functools import lru_cache
import asyncio
import json

logger = get_logger(__name__)

MAX_CANDIDATES = 10  # 只对前 10 个候选做 re-ranking


async def rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
    method: str | None = None,
) -> list[RetrievalResult]:
    """对检索结果做二次排序。method 不传时使用配置 rerank_method（cross_encoder / llm）。"""
    method = method or get_settings().rerank_method
    if method == "llm":
        return await llm_rerank(question, results, top_k)
    return await cross_encoder_rerank(question, results, top_k)


# ── Cross-encoder ──


@lru_cache
def _load_cross_encoder(model_name: str):
    """加载一次后常驻内存。首次运行会从 Hugging Face 下载模型（bge-reranker-base 约 1.1GB）。"""
    from sentence_transformers import CrossEncoder

    logger.info("reranker.cross_encoder.loading", model=model_name)
    return CrossEncoder(model_name, max_length=512)


async def cross_encoder_rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """用 Cross-encoder 对检索结果做二次排序。

    和 Embedding（问题、文档各自编码成向量再比较）不同，Cross-encoder 把"问题 + 文档"拼在一起
    输入模型，直接输出相关性分数，更准，但每个候选都要算一遍，所以只用于少量候选的精排。
    模型在本地 CPU 上运行，不需要 API 调用。
    """
    if len(results) <= 1:
        return results

    candidates = results[:MAX_CANDIDATES]
    pairs = [(question, r.content) for r in candidates]

    try:
        model = _load_cross_encoder(get_settings().cross_encoder_model)
        # predict 是 CPU 密集的同步计算，放到线程里执行，避免阻塞事件循环里的其他请求
        scores = await asyncio.to_thread(model.predict, pairs)

        for result, score in zip(candidates, scores):
            result.score = round(float(score), 4)

        reranked = sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]
        logger.info(
            "reranker.cross_encoder.completed",
            candidates=len(candidates),
            top_score=reranked[0].score if reranked else 0,
        )
        return reranked

    except Exception as e:
        logger.warning("reranker.cross_encoder.failed", error=str(e))
        return results[:top_k]  # 降级：不 re-rank，直接取前 K


# ── LLM ──

RERANK_PROMPT = """You are a relevance judge. Given a user's question and a list of text passages, rate how relevant each passage is to answering the question.

For each passage, output a relevance score from 0.0 (completely irrelevant) to 1.0 (directly answers the question).

User's question:{question}

Passages:
{passages}

Output ONLY a JSON array of scores in the same order as the passages, nothing else.
Example: [0.9, 0.3, 0.7, 0.1, 0.8]"""


async def llm_rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """用 LLM 对检索结果做二次排序。"""

    if len(results) <= 1:
        return results

    # 只对前 10 个结果做 re-ranking（控制成本）
    candidates = results[:MAX_CANDIDATES]

    passages = "\n\n".join(
        f"[Passage{i+1}]:{r.content[:300]}"
        for i, r in enumerate(candidates)
    )

    prompt = RERANK_PROMPT.format(
        question=question,
        passages=passages,
    )

    try:
        adapter = get_llm_adapter()
        response = await adapter.generate([{"role": "user", "content": prompt}])

        scores = json.loads(response.strip())

        if len(scores) != len(candidates):
            logger.warning("reranker.score_mismatch", expected=len(candidates), got=len(scores))
            return results[:top_k]

        for result, score in zip(candidates, scores):
            result.score = float(score)

        reranked = sorted(candidates, key=lambda r: r.score, reverse=True)[:top_k]

        logger.info(
            "reranker.completed",
            candidates=len(candidates),
            top_score=reranked[0].score if reranked else 0,
        )

        return reranked

    except Exception as e:
        logger.warning("reranker.failed", error=str(e))
        return results[:top_k]  # 降级：不 re-rank，直接取前 K