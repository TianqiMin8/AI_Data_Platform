"""作业 6-4：文件上传去重。

使用本地数据库，所有写入在事务里完成后回滚；process_document 换成假的，不调用 OpenAI。
"""

import asyncio
import hashlib
import io

import pytest
from starlette.datastructures import UploadFile

import app.services.document as doc_service
from app.db.session import async_session_factory, engine
from app.models.user import User

CONTENT = "第一段：销售数据概览。\n\n第二段：售后政策。".encode("utf-8")


def _upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=name)


class FakePipeline:
    """记录被处理了几次；可以让下一次处理失败。"""

    def __init__(self):
        self.calls = 0
        self.fail_next = False

    async def __call__(self, db, document, file_path):
        self.calls += 1
        document.status = "failed" if self.fail_next else "completed"
        self.fail_next = False
        await db.flush()
        return document


def _run(scenario):
    """在一个会回滚的事务里跑测试场景，传入两个用户。"""

    async def main():
        async with async_session_factory() as db:
            try:
                users = []
                for i in range(2):
                    user = User(email=f"dedup{i}@test.local", hashed_password="x", nickname=f"u{i}")
                    db.add(user)
                    users.append(user)
                await db.flush()
                await scenario(db, *users)
            finally:
                await db.rollback()
        # 应用的 engine 是全局的；每个测试用 asyncio.run 新建事件循环，
        # 连接池里的连接属于上一个循环，必须释放掉，否则下一个测试会报 "Event loop is closed"
        await engine.dispose()

    try:
        asyncio.run(main())
    except OSError:
        pytest.skip("本地数据库不可用")


@pytest.fixture
def pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(doc_service, "process_document", fake)
    return fake


def test_stores_sha256_of_file_bytes(pipeline):
    async def scenario(db, alice, bob):
        doc, is_dup = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        assert not is_dup
        assert doc.content_hash == hashlib.sha256(CONTENT).hexdigest()

    _run(scenario)


def test_same_content_returns_existing_document(pipeline):
    async def scenario(db, alice, bob):
        first, _ = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        second, is_dup = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        assert is_dup
        assert second.id == first.id
        assert pipeline.calls == 1  # 第二次没有重新处理

    _run(scenario)


def test_different_filename_same_content_is_duplicate(pipeline):
    async def scenario(db, alice, bob):
        first, _ = await doc_service.upload_document(db, alice, _upload("销售报告.txt", CONTENT))
        second, is_dup = await doc_service.upload_document(db, alice, _upload("副本.txt", CONTENT))
        assert is_dup
        assert second.id == first.id
        assert second.filename == "销售报告.txt"  # 返回的是已有文档，保留原文件名

    _run(scenario)


def test_one_byte_change_is_not_duplicate(pipeline):
    async def scenario(db, alice, bob):
        first, _ = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        edited = CONTENT.replace("售后".encode(), "售前".encode())
        second, is_dup = await doc_service.upload_document(db, alice, _upload("a.txt", edited))
        assert not is_dup
        assert second.id != first.id
        assert second.content_hash != first.content_hash
        assert pipeline.calls == 2

    _run(scenario)


def test_other_users_document_is_not_reused(pipeline):
    async def scenario(db, alice, bob):
        alice_doc, _ = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        bob_doc, is_dup = await doc_service.upload_document(db, bob, _upload("a.txt", CONTENT))
        assert not is_dup
        assert bob_doc.id != alice_doc.id
        assert bob_doc.user_id == bob.id

    _run(scenario)


def test_failed_document_is_reprocessed(pipeline):
    async def scenario(db, alice, bob):
        pipeline.fail_next = True
        failed, _ = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        assert failed.status == "failed"

        retried, is_dup = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        assert not is_dup
        assert retried.id != failed.id
        assert retried.status == "completed"
        assert pipeline.calls == 2

        # 之后再传，复用的是成功的那份
        again, is_dup = await doc_service.upload_document(db, alice, _upload("a.txt", CONTENT))
        assert is_dup
        assert again.id == retried.id

    _run(scenario)
