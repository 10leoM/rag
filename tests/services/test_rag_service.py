# -*- coding: utf-8 -*-
"""F4 · RAG 检索服务测试：三种模式与混合融合。"""

import json
from pathlib import Path

import pytest_asyncio

from app.infrastructure.embedding import LocalEmbedder
from app.infrastructure.vectordb import MilvusManager, MilvusRecord
from app.services import RAGService

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "rag"


@pytest_asyncio.fixture
async def chunks_data() -> dict:
    """加载 20 个分块与 5 个带 ground-truth 的查询。"""
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
    """准备数据：embedding -> 入库 -> 加载 -> 注册 BM25。"""
    chunks = chunks_data["chunks"]
    texts = [c["text"] for c in chunks]

    # 1. 向量化
    vectors = await embedder.aembed_documents(texts)

    # 2. 组装记录并写入 Milvus
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

    # 3. 加载集合并构造服务
    col = milvus.get_collection(collection)
    col.load()
    service = RAGService(embedder, col)

    # 4. 注册 BM25 关键词索引
    service.register_documents({c["id"]: c["text"] for c in chunks})
    return service


def _recall_at_k(results, ground_truth: list[str], k: int = 5) -> float:
    """ground-truth 中落入 top-k 的比例。"""
    result_ids = {r.id for r in results[:k]}
    hit = len(result_ids.intersection(ground_truth))
    return hit / len(ground_truth)


async def test_retrieve_modes_and_hybrid(
    rag_service: RAGService,
    chunks_data: dict,
) -> None:
    """三种模式均能召回 ground-truth，且混合不低于单路。"""
    for item in chunks_data["queries"]:
        query = item["query"]
        ground_truth = item["ground_truth"]

        vector = await rag_service.retrieve(query, mode="vector", top_k=5)
        keyword = await rag_service.retrieve(query, mode="keyword", top_k=5)
        hybrid = await rag_service.retrieve(query, mode="hybrid", top_k=5)

        vr = _recall_at_k(vector, ground_truth, 5)
        kr = _recall_at_k(keyword, ground_truth, 5)
        hr = _recall_at_k(hybrid, ground_truth, 5)

        assert vr > 0, f"vector 未召回 ground-truth: {query}"
        assert hr >= vr, f"hybrid 低于 vector: {query}"
        assert hr >= kr, f"hybrid 低于 keyword: {query}"
