"""作业 7-4：Cross-encoder re-ranking。用假模型，不下载真实模型、不调用 LLM。"""

import asyncio

import pytest

import app.rag.reranker as reranker
from app.rag.retriever import RetrievalResult


def _results(*contents):
    return [RetrievalResult(chunk_id=i, content=c, score=0.5) for i, c in enumerate(contents)]


class FakeCrossEncoder:
    """内容里含"客服"的给高分，其余给低分；记录收到的输入。"""

    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [0.9 if "客服" in passage else 0.1 * (i % 3) for i, (_, passage) in enumerate(pairs)]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeCrossEncoder()
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda name: model)
    return model


def test_reorders_by_cross_encoder_score(fake_model):
    results = _results("当日达城市", "积分规则", "人工客服早八点到晚十点")
    reranked = asyncio.run(reranker.cross_encoder_rerank("晚上能找客服吗", results, top_k=2))
    assert reranked[0].content == "人工客服早八点到晚十点"
    assert reranked[0].score == 0.9
    assert len(reranked) == 2


def test_sends_question_passage_pairs(fake_model):
    asyncio.run(reranker.cross_encoder_rerank("问题", _results("A", "B"), top_k=2))
    assert fake_model.calls == [[("问题", "A"), ("问题", "B")]]


def test_only_top_10_candidates_are_scored(fake_model):
    asyncio.run(reranker.cross_encoder_rerank("问题", _results(*[f"段落{i}" for i in range(15)]), top_k=3))
    assert len(fake_model.calls[0]) == 10


def test_single_result_is_returned_as_is(fake_model):
    results = _results("唯一的结果")
    assert asyncio.run(reranker.cross_encoder_rerank("问题", results)) == results
    assert fake_model.calls == []


def test_model_failure_falls_back_to_original_order(monkeypatch):
    def broken(name):
        raise OSError("模型下载失败")
    monkeypatch.setattr(reranker, "_load_cross_encoder", broken)

    results = _results("A", "B", "C", "D")
    reranked = asyncio.run(reranker.cross_encoder_rerank("问题", results, top_k=2))
    assert [r.content for r in reranked] == ["A", "B"]


def test_dispatch_uses_setting_and_override(fake_model, monkeypatch):
    called = []

    async def fake_llm(question, results, top_k):
        called.append("llm")
        return results[:top_k]

    monkeypatch.setattr(reranker, "llm_rerank", fake_llm)
    results = _results("A", "人工客服")

    monkeypatch.setattr(reranker.get_settings(), "rerank_method", "cross_encoder")
    asyncio.run(reranker.rerank("问题", results))
    assert called == [] and len(fake_model.calls) == 1

    asyncio.run(reranker.rerank("问题", results, method="llm"))  # 请求里指定的方法优先于配置
    assert called == ["llm"]
