"""作业 7-4：对比 LLM re-ranking 和 Cross-encoder re-ranking 的速度和效果。

每题先用向量检索取同一批 Top-10 候选，再分别交给两种 re-ranker 排出 Top-3，
和"不 re-rank（直接用向量检索的顺序）"一起比较 Hit@1 / Hit@3 / MRR@3 和耗时。
测试文档和题目复用 7-3 的 scripts/eval_data/。

用法：
    poetry run python scripts/compare_rerankers.py
"""

import asyncio
import copy
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_retrieval import DATA_DIR, label_questions, load_document  # noqa: E402  (同时完成了日志静音)

import json  # noqa: E402

from app.config.settings import get_settings  # noqa: E402
from app.db.session import async_session_factory, engine  # noqa: E402
from app.models.user import User  # noqa: E402
from app.rag.reranker import _load_cross_encoder, cross_encoder_rerank, llm_rerank  # noqa: E402
from app.rag.retriever import vector_search  # noqa: E402

CANDIDATES = 10
TOP_K = 3


async def no_rerank(question, results, top_k):
    return results[:top_k]


METHODS = [("不 re-rank", no_rerank), ("LLM", llm_rerank), ("Cross-encoder", cross_encoder_rerank)]


async def main(user_id: int = 1):
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))

    start = time.perf_counter()
    _load_cross_encoder(get_settings().cross_encoder_model)
    print(f"Cross-encoder 模型加载（只在进程启动后第一次发生）：{time.perf_counter() - start:.1f}s")

    async with async_session_factory() as db:
        user = await db.get(User, user_id)
        doc = await load_document(db, user)
        await label_questions(db, doc, questions)

        ranks = {name: [] for name, _ in METHODS}
        latency = {name: [] for name, _ in METHODS}
        top1 = {name: [] for name, _ in METHODS}

        for q in questions:
            candidates = await vector_search(
                db, q["question"], user.id, top_k=CANDIDATES, document_ids=[doc.id], score_threshold=0.0
            )
            for name, fn in METHODS:
                start = time.perf_counter()
                ranked = await fn(q["question"], copy.deepcopy(candidates), TOP_K)
                latency[name].append((time.perf_counter() - start) * 1000)
                ids = [r.chunk_id for r in ranked]
                ranks[name].append(ids.index(q["answer_chunk_id"]) + 1 if q["answer_chunk_id"] in ids else None)
                top1[name].append(ids[0] if ids else None)

    await engine.dispose()

    total = len(questions)
    print(f"\n每题候选数 {CANDIDATES}，取 Top-{TOP_K}，共 {total} 题\n")
    print(f"{'方法':<16}{'Hit@1':>8}{'Hit@3':>8}{'MRR@3':>8}{'平均耗时':>12}{'中位数':>10}{'最慢':>10}")
    for name, _ in METHODS:
        r = ranks[name]
        hit1 = sum(x == 1 for x in r)
        hit3 = sum(x is not None for x in r)
        mrr = sum(1 / x for x in r if x) / total
        lat = latency[name]
        print(
            f"{name:<14}{hit1:>6}/{total}{hit3:>6}/{total}{mrr:>8.2f}"
            f"{statistics.mean(lat):>10.0f}ms{statistics.median(lat):>8.0f}ms{max(lat):>8.0f}ms"
        )

    agree = sum(a == b for a, b in zip(top1["LLM"], top1["Cross-encoder"]))
    print(f"\nLLM 和 Cross-encoder 排在第 1 名的 chunk 相同：{agree}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
