"""作业 6-3：chunk 的 token 计数。"""

import asyncio

import tiktoken

import app.rag.pipeline as pipeline
from app.models.document import Document
from app.rag.embedder import EMBEDDING_MODEL, count_embedding_tokens
from tests.test_chunk_dedup import FakeEmbedder, FakeSession


def test_uses_embedding_model_encoding():
    # text-embedding-3-small 是 cl100k_base，和 gpt-4o 的 o200k_base 不同
    assert tiktoken.encoding_for_model(EMBEDDING_MODEL).name == "cl100k_base"
    text = "每个品类的销售额是多少？"
    assert count_embedding_tokens(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_empty_text_has_zero_tokens():
    assert count_embedding_tokens("") == 0


def test_chinese_uses_more_tokens_per_char_than_english():
    english = "The total sales amount for each product category this quarter."
    chinese = "本季度每个产品品类的销售总额是多少，请按金额从高到低排序。"
    en_ratio = count_embedding_tokens(english) / len(english)
    zh_ratio = count_embedding_tokens(chinese) / len(chinese)
    assert zh_ratio > en_ratio


def test_pipeline_fills_token_count(tmp_path, monkeypatch):
    text = "第一段：销售数据概览。\n\n" + "The quarterly revenue grew by 15 percent. " * 20
    path = tmp_path / "doc.txt"
    path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(pipeline, "get_embedder", lambda: FakeEmbedder())
    db = FakeSession()
    document = Document(id=1, user_id=1, filename="doc.txt", file_type="txt", file_size=len(text))

    asyncio.run(pipeline.process_document(db, document, str(path)))

    assert db.added
    for chunk in db.added:
        assert chunk.token_count == count_embedding_tokens(chunk.content)
        assert chunk.token_count > 0
