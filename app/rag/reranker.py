from app.llm.factory import get_llm_adapter
from app.rag.retriever import RetrievalResult
from app.core.logging import get_logger
import json

logger = get_logger(__name__)

RERANK_PROMPT = """You are a relevance judge. Given a user's question and a list of text passages, rate how relevant each passage is to answering the question.

For each passage, output a relevance score from 0.0 (completely irrelevant) to 1.0 (directly answers the question).

User's question:{question}

Passages:
{passages}

Output ONLY a JSON array of scores in the same order as the passages, nothing else.
Example: [0.9, 0.3, 0.7, 0.1, 0.8]"""


async def rerank(
    question: str,
    results: list[RetrievalResult],
    top_k: int = 3,
) -> list[RetrievalResult]:
    """用 LLM 对检索结果做二次排序。"""

    if len(results) <= 1:
        return results

    # 只对前 10 个结果做 re-ranking（控制成本）
    candidates = results[:10]

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