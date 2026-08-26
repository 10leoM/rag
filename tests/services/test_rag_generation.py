# -*- coding: utf-8 -*-
"""F5 · 重排与生成测试：重排召回、引用解析。"""

import json
from pathlib import Path

import pytest_asyncio

from app.infrastructure.embedding import LocalEmbedder
from app.infrastructure.vectordb import MilvusManager, MilvusRecord
from app.services import RAGService

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "rag"


class FakeLLM:
    """固定答案的假 LLM，用于验证引用解析，不依赖 OpenAI。"""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def ainvoke(self, messages, **kwargs) -> str:
        return self._answer


@pytest_asyncio.fixture
async def chunks_data() -> dict:
    path = DATA_DIR / "chunks.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest_asyncio.fixture
async def rag_service(
    embedder: LocalEmbedder,
    milvus: MilvusManager,
    collection: str,
    chunks_data: dict,
) -> RAGService:
    """准备检索数据并注入 FakeLLM 构造服务。"""
    chunks = chunks_data["chunks"]
    texts = [c["text"] for c in chunks]
    vectors = await embedder.aembed_documents(texts)
    records = [
        MilvusRecord(
            id=c["id"],
            embedding=vec,
            text=c["text"],
            document_id="rag-test",
            chunk_index=i,
            filename="chunks.json",
            mime_type="application/json",
            metadata={},
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]
    await milvus.insert(collection, records)
    col = milvus.get_collection(collection)
    col.load()

    llm = FakeLLM("根据上下文，有限责任公司股东人数上限为五十人。[1]")
    service = RAGService(embedder, col, llm=llm)
    service.register_documents({c["id"]: c["text"] for c in chunks})
    return service


def _recall_at_k(results, ground_truth: list[str], k: int = 5) -> float:
    result_ids = {r.id for r in results[:k]}
    hit = len(result_ids.intersection(ground_truth))
    return hit / len(ground_truth)


async def test_rerank_improves_or_keeps_recall(
    rag_service: RAGService,
    chunks_data: dict,
) -> None:
    """重排后 recall@5 不低于重排前。"""
    for item in chunks_data["queries"]:
        query = item["query"]
        ground_truth = item["ground_truth"]

        results = await rag_service.retrieve(query, mode="hybrid", top_k=10)
        before = _recall_at_k(results, ground_truth, 5)

        reranked = await rag_service.rerank(query, results, top_k=5)
        after = _recall_at_k(reranked, ground_truth, 5)

        assert after >= before, f"重排后 recall 下降: {query}"


async def test_answer_citations(
    rag_service: RAGService,
    chunks_data: dict,
) -> None:
    """答案含引用，引用映射到上下文且无越界。"""
    query = chunks_data["queries"][0]["query"]
    response = await rag_service.answer(query, mode="hybrid", top_k=10, rerank_top_k=5)

    assert "[1]" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].index == 1
    assert response.citations[0].result_id == response.raw_contexts[0].id
