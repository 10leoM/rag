# -*- coding: utf-8 -*-
"""F6 · RAG 查询 API 测试：请求/响应结构与 trace。"""

from fastapi.testclient import TestClient

from app.api.routes.rag import get_rag_service
from app.main import app
from app.models.schemas import (
    Citation,
    RAGQueryRequest,
    RAGResponse,
    RetrievalResult,
)


class FakeRAGService:
    """固定响应的假服务，聚焦验证 API 层序列化。"""

    async def answer(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        rerank_top_k: int = 5,
        include_trace: bool = False,
    ) -> RAGResponse:
        contexts = [
            RetrievalResult(
                id="c01",
                content="有限责任公司由五十个以下股东出资设立。",
                score=0.9,
                source="hybrid",
            )
        ]
        response = RAGResponse(
            answer="有限责任公司由五十个以下股东出资设立。[1]",
            citations=[Citation(index=1, result_id="c01", snippet=contexts[0].content)],
            raw_contexts=contexts,
            model="openai",
        )
        if include_trace:
            response.trace = {
                "retrieved": contexts,
                "reranked": contexts,
            }
        return response


def test_rag_query_without_trace() -> None:
    """不带 trace 时返回结构化响应，且无 trace 字段。"""
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    client = TestClient(app)
    try:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "有限责任公司股东人数上限是多少", "include_trace": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert data["citations"]
        assert data["raw_contexts"]
        assert data["raw_contexts"][0]["score"] is not None
        assert "trace" not in data or data["trace"] is None
    finally:
        app.dependency_overrides.clear()


def test_rag_query_with_trace() -> None:
    """带 trace 时返回检索与重排中间态。"""
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    client = TestClient(app)
    try:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "有限责任公司股东人数上限是多少", "include_trace": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace"] is not None
        assert "retrieved" in data["trace"]
        assert "reranked" in data["trace"]
        assert len(data["trace"]["retrieved"]) == 1
    finally:
        app.dependency_overrides.clear()
