"""作业 7-3：检索质量评估（Hit@3）。

流程：
1. 上传测试文档 scripts/eval_data/handbook.md（已上传过会被 6-4 的去重直接复用，不重复调用 Embedding）
2. 按 questions.json 中每题的 answer_contains（答案 chunk 里必须出现的一句原文）找到"正确 chunk"
   —— 不直接写死 chunk ID：每次重新上传 ID 都会变
3. 每题分别用 vector / bm25 / hybrid / hybrid+rerank 检索 Top-3，统计正确 chunk 是否在其中

用法：
    poetry run python scripts/eval_retrieval.py              # 评估后保留测试文档，下次运行更快
    poetry run python scripts/eval_retrieval.py --cleanup    # 评估后删除测试文档
    poetry run python scripts/eval_retrieval.py --user-id 2  # 用其他用户的身份（默认 1）

会调用 OpenAI：首次上传时生成 Embedding；每次运行生成查询向量（有 7-1 的缓存）；
hybrid+rerank 每题调用一次 LLM。
"""

import argparse
import asyncio
import io
import json
import logging
import sys
from pathlib import Path

import structlog

# 只看评估结果：关掉 SQL 回显和 info 级别的业务日志
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from sqlalchemy import select  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402

from app.db.session import async_session_factory, engine  # noqa: E402
from app.models.document import Chunk  # noqa: E402
from app.models.user import User  # noqa: E402
from app.schemas.search import SearchRequest  # noqa: E402
from app.services import document as doc_service  # noqa: E402
from app.services.search import search_documents  # noqa: E402

engine.echo = False

DATA_DIR = Path(__file__).parent / "eval_data"
TOP_K = 3
CONFIGS = [  # (显示名, search_method, 是否 rerank)
    ("vector", "vector", False),
    ("bm25", "bm25", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


async def load_document(db, user):
    content = (DATA_DIR / "handbook.md").read_bytes()
    upload = UploadFile(file=io.BytesIO(content), filename="eval_handbook.md")
    doc, is_duplicate = await doc_service.upload_document(db, user, upload)
    await db.commit()
    if doc.status != "completed":
        sys.exit(f"测试文档处理失败：{doc.error_message}")
    print(f"测试文档 id={doc.id}，{doc.chunk_count} 个 chunk（{'复用已上传的' if is_duplicate else '新上传'}）")
    return doc


async def label_questions(db, doc, questions):
    """把每题的 answer_contains 映射成正确 chunk 的 ID。"""
    chunks = (await db.execute(select(Chunk).where(Chunk.document_id == doc.id))).scalars().all()
    for q in questions:
        matched = [c.id for c in chunks if q["answer_contains"] in c.content]
        if len(matched) != 1:
            sys.exit(f"标注有误：『{q['answer_contains']}』匹配到 {len(matched)} 个 chunk，应该恰好 1 个")
        q["answer_chunk_id"] = matched[0]


async def evaluate(db, user, doc, questions):
    results = {name: [] for name, _, _ in CONFIGS}  # 每题的命中排名（1~3），未命中为 None
    for q in questions:
        for name, method, rerank in CONFIGS:
            req = SearchRequest(
                query=q["question"], top_k=TOP_K, search_method=method,
                rerank=rerank, document_ids=[doc.id],
            )
            response = await search_documents(db, user, req)
            ids = [r.chunk_id for r in response.results]
            rank = ids.index(q["answer_chunk_id"]) + 1 if q["answer_chunk_id"] in ids else None
            results[name].append(rank)
    return results


def report(questions, results):
    names = [name for name, _, _ in CONFIGS]
    print(f"\n{'#':<3}{'类型':<8}{'问题':<26}" + "".join(f"{n:>15}" for n in names))
    for i, q in enumerate(questions):
        cells = "".join(f"{(f'✓ 第{r}名' if r else '✗'):>14}" for r in (results[n][i] for n in names))
        print(f"{i + 1:<3}{q['type']:<6}{q['question'][:18]:<22}{cells}")

    total = len(questions)
    print(f"\nHit@{TOP_K}（正确 chunk 出现在前 {TOP_K} 条结果中的比例）：")
    for name in names:
        hits = sum(r is not None for r in results[name])
        # MRR：命中排名的倒数取平均，第 1 名 = 1，第 2 名 = 0.5，未命中 = 0
        mrr = sum(1 / r for r in results[name] if r) / total
        print(f"  {name:<15} {hits}/{total} = {hits / total:>4.0%}    MRR@{TOP_K} = {mrr:.2f}")

    print("\n按问题类型：")
    for qtype in dict.fromkeys(q["type"] for q in questions):
        idx = [i for i, q in enumerate(questions) if q["type"] == qtype]
        cells = "  ".join(f"{n} {sum(results[n][i] is not None for i in idx)}/{len(idx)}" for n in names)
        print(f"  {qtype:<6} {cells}")


async def main(user_id: int, cleanup: bool):
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    async with async_session_factory() as db:
        user = await db.get(User, user_id)
        if not user:
            sys.exit(f"用户 {user_id} 不存在")

        doc = await load_document(db, user)
        await label_questions(db, doc, questions)
        results = await evaluate(db, user, doc, questions)
        report(questions, results)

        if cleanup:
            await doc_service.delete_document(db, user, doc.id)
            await db.commit()
            print(f"\n已删除测试文档 id={doc.id}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索质量评估（Hit@3）")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--cleanup", action="store_true", help="评估后删除测试文档")
    args = parser.parse_args()
    asyncio.run(main(args.user_id, args.cleanup))
