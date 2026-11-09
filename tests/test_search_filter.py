"""作业 7-2：按文档过滤的检索。

使用本地数据库（pgvector 的 <=> 和 PostgreSQL 全文检索没法用假数据库模拟），
数据在事务里创建、测试后回滚；查询向量用假 Embedder 生成，不调用 OpenAI。
"""

import asyncio

import pytest

import app.rag.retriever as retriever
from app.core.errors import NotFoundError
from app.db.session import async_session_factory, engine
from app.models.document import Chunk, Document
from app.models.user import User
from app.schemas.search import SearchRequest
from app.services.search import search_documents

VECTOR = [0.01] * 1536  # 所有 chunk 和查询用同一个向量：不加过滤时两个文档都会命中


class FakeEmbedder:
    async def embed_query(self, query):
        return VECTOR


def _run(scenario):
    async def main():
        async with async_session_factory() as db:
            try:
                alice = User(email="search-alice@test.local", hashed_password="x", nickname="alice")
                bob = User(email="search-bob@test.local", hashed_password="x", nickname="bob")
                db.add_all([alice, bob])
                await db.flush()

                docs = {}
                for owner, name, text in [
                    (alice, "policy_a.txt", "refund policy: customers can return items within seven days"),
                    (alice, "policy_b.txt", "refund policy: products come with a one year warranty"),
                    (bob, "policy_bob.txt", "refund policy: bob's private document"),
                ]:
                    doc = Document(user_id=owner.id, filename=name, file_type="txt", file_size=len(text), status="completed")
                    db.add(doc)
                    await db.flush()
                    db.add(Chunk(document_id=doc.id, chunk_index=0, content=text, char_count=len(text), embedding=VECTOR))
                    docs[name] = doc
                await db.flush()

                await scenario(db, alice, docs)
            finally:
                await db.rollback()
        await engine.dispose()

    try:
        asyncio.run(main())
    except OSError:
        pytest.skip("本地数据库不可用")


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(retriever, "get_embedder", lambda: FakeEmbedder())


def _search(db, user, method, document_ids=None):
    req = SearchRequest(query="refund policy", top_k=10, search_method=method, rerank=False, document_ids=document_ids)
    return search_documents(db, user, req)


@pytest.mark.parametrize("method", ["vector", "bm25", "hybrid"])
def test_without_filter_searches_all_own_documents(method):
    async def scenario(db, alice, docs):
        response = await _search(db, alice, method)
        found = {r.document_id for r in response.results}
        assert found == {docs["policy_a.txt"].id, docs["policy_b.txt"].id}  # 不包括 bob 的文档

    _run(scenario)


@pytest.mark.parametrize("method", ["vector", "bm25", "hybrid"])
def test_filter_returns_only_specified_document(method):
    async def scenario(db, alice, docs):
        target = docs["policy_a.txt"].id
        response = await _search(db, alice, method, [target])
        assert response.results
        assert {r.document_id for r in response.results} == {target}

    _run(scenario)


def test_empty_list_means_no_filter():
    async def scenario(db, alice, docs):
        response = await _search(db, alice, "bm25", [])
        assert len({r.document_id for r in response.results}) == 2

    _run(scenario)


def test_duplicate_ids_are_fine():
    async def scenario(db, alice, docs):
        target = docs["policy_b.txt"].id
        response = await _search(db, alice, "vector", [target, target])
        assert {r.document_id for r in response.results} == {target}

    _run(scenario)


def test_other_users_document_id_is_rejected():
    async def scenario(db, alice, docs):
        bob_doc = docs["policy_bob.txt"].id
        with pytest.raises(NotFoundError) as exc:
            await _search(db, alice, "vector", [docs["policy_a.txt"].id, bob_doc])
        assert str(bob_doc) in exc.value.message

    _run(scenario)


def test_nonexistent_document_id_is_rejected():
    async def scenario(db, alice, docs):
        with pytest.raises(NotFoundError):
            await _search(db, alice, "bm25", [999_999_999])

    _run(scenario)
