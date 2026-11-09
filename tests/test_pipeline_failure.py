"""Pipeline 失败时：不抛异常、文档标记为 failed 并保留、已写入的 chunk 被撤销。"""

import asyncio

import pytest
from sqlalchemy import func, select

import app.rag.pipeline as pipeline
from app.db.session import async_session_factory, engine
from app.models.document import Chunk, Document
from app.models.user import User
from tests.test_chunk_dedup import FakeEmbedder, FakeSession

TEXT = "第一段：销售数据概览。\n\n第二段：售后政策，购买后七天内可无理由退货。"


class BrokenEmbedder(FakeEmbedder):
    async def embed_batch(self, texts):
        raise RuntimeError("OpenAI 429 rate limit")


def _write(tmp_path, text=TEXT, name="doc.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── 用假 session：行为层面 ──


def test_parse_failure_returns_failed_document_without_raising(tmp_path):
    db = FakeSession()
    document = Document(id=1, user_id=1, filename="doc.xyz", file_type="xyz", file_size=1)

    result = asyncio.run(pipeline.process_document(db, document, _write(tmp_path)))

    assert result is document
    assert document.status == "failed"
    assert document.error_message.startswith("[parsing]")
    assert "不支持的文件类型" in document.error_message


def test_blank_document_fails_at_chunking(tmp_path):
    # TxtParser 会返回一页空白内容，到切片阶段才发现没有有效内容
    db = FakeSession()
    document = Document(id=1, user_id=1, filename="empty.txt", file_type="txt", file_size=0)
    asyncio.run(pipeline.process_document(db, document, _write(tmp_path, text="   ")))
    assert document.status == "failed"
    assert document.error_message.startswith("[chunking]")


def test_embedding_failure_discards_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_embedder", lambda: BrokenEmbedder())
    db = FakeSession()
    document = Document(id=1, user_id=1, filename="doc.txt", file_type="txt", file_size=1)

    asyncio.run(pipeline.process_document(db, document, _write(tmp_path)))

    assert document.status == "failed"
    assert document.error_message == "[embedding] OpenAI 429 rate limit"
    assert document.chunk_count == 0
    assert db.added == []  # SAVEPOINT 回滚：已经 add 的 chunk 被撤销


# ── 真实数据库：SAVEPOINT 确实生效 ──


def test_failed_document_is_kept_and_chunks_rolled_back_in_db(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "get_embedder", lambda: BrokenEmbedder())
    path = _write(tmp_path)

    async def main():
        async with async_session_factory() as db:
            try:
                user = User(email="pipeline-fail@test.local", hashed_password="x", nickname="u")
                db.add(user)
                await db.flush()
                document = Document(user_id=user.id, filename="doc.txt", file_type="txt", file_size=10)
                db.add(document)
                await db.flush()

                result = await pipeline.process_document(db, document, path)

                # 返回后属性可以直接访问（没有因 SAVEPOINT 回滚而触发异步懒加载报错）
                assert result.filename == "doc.txt"
                assert result.status == "failed"

                # 数据库里：文档还在、状态是 failed；chunk 已经插入过，但随 SAVEPOINT 一起撤销了
                row = (await db.execute(select(Document).where(Document.id == document.id))).scalar_one()
                assert row.status == "failed"
                assert row.error_message.startswith("[embedding]")
                chunk_count = await db.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
                )
                assert chunk_count == 0
            finally:
                await db.rollback()
        await engine.dispose()

    try:
        asyncio.run(main())
    except OSError:
        pytest.skip("本地数据库不可用")
