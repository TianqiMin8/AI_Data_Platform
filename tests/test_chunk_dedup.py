"""作业 6-2：chunk 去重。"""

import asyncio

import app.rag.pipeline as pipeline
from app.models.document import Document
from app.rag.chunker import ChunkResult, content_hash, deduplicate_chunks


def _chunk(content: str, index: int = 0, section: str | None = None) -> ChunkResult:
    return ChunkResult(content=content, chunk_index=index, char_count=len(content), section_title=section)


# ── 去重函数 ──


def test_hash_is_sha256_hex():
    assert content_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_removes_exact_duplicates_and_keeps_first():
    chunks = [_chunk("A", 0, "第一章"), _chunk("B", 1), _chunk("A", 2, "第三章"), _chunk("C", 3), _chunk("B", 4)]
    unique, removed = deduplicate_chunks(chunks)
    assert [c.content for c in unique] == ["A", "B", "C"]
    assert removed == 2
    assert unique[0].section_title == "第一章"  # 保留第一次出现的那个（连同它的 metadata）


def test_no_duplicates():
    chunks = [_chunk("A"), _chunk("B")]
    unique, removed = deduplicate_chunks(chunks)
    assert unique == chunks
    assert removed == 0


def test_only_exact_match_counts():
    # 只差一个字、一个标点或一个空格，都不算重复
    chunks = [_chunk("销售额为 100 元。"), _chunk("销售额为 100 元"), _chunk("销售额为 200 元。"), _chunk("销售额为100 元。")]
    unique, removed = deduplicate_chunks(chunks)
    assert removed == 0
    assert len(unique) == 4


def test_empty_list():
    assert deduplicate_chunks([]) == ([], 0)


# ── Pipeline：重复的 chunk 不入库、不做 Embedding ──


class FakeSession:
    """只记录 add_all 的内容。begin_nested 失败时模拟 SAVEPOINT 回滚：丢弃其中 add 的记录。"""

    def __init__(self):
        self.added = []

    def add_all(self, records):
        self.added.extend(records)

    async def flush(self):
        pass

    async def refresh(self, obj):
        pass

    def begin_nested(self):
        session = self

        class Savepoint:
            async def __aenter__(self):
                self.mark = len(session.added)

            async def __aexit__(self, exc_type, exc, tb):
                if exc_type:
                    del session.added[self.mark:]
                return False

        return Savepoint()


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self):
        self.embedded_texts: list[str] = []

    async def embed_batch(self, texts):
        self.embedded_texts.extend(texts)
        return [[0.0] * 3 for _ in texts]


def _paragraph(topic: str) -> str:
    # 每段约 300 字：两段合起来超过 chunk_size=500，所以每段单独成为一个 chunk
    return f"{topic}。" + "这是用于测试去重的正文内容，长度足够单独成为一个切片。" * 11


def test_pipeline_skips_duplicate_chunks(tmp_path, monkeypatch):
    disclaimer = _paragraph("免责声明：本文档内容仅供内部参考")
    text = "\n\n".join([
        _paragraph("第一部分介绍销售数据"),
        disclaimer,
        _paragraph("第二部分介绍售后政策"),
        disclaimer,
        disclaimer,
    ])
    path = tmp_path / "doc.txt"
    path.write_text(text, encoding="utf-8")

    embedder = FakeEmbedder()
    monkeypatch.setattr(pipeline, "get_embedder", lambda: embedder)
    db = FakeSession()
    document = Document(id=1, user_id=1, filename="doc.txt", file_type="txt", file_size=len(text))

    asyncio.run(pipeline.process_document(db, document, str(path)))

    assert document.status == "completed"
    assert document.chunk_count == 3                      # 5 个 chunk，去掉 2 个重复
    assert [c.chunk_index for c in db.added] == [0, 1, 2]  # 编号连续，没有空洞
    assert sum(c.content == disclaimer for c in db.added) == 1
    assert len(embedder.embedded_texts) == 3               # 重复内容没有被送去生成 Embedding
