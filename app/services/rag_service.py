# -*- coding: utf-8 -*-
"""RAG 检索服务：封装多路检索（向量 / BM25 / 混合）。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.rag.retriever import MultiRetriever
from app.infrastructure.embedding import LocalEmbedder
from app.models.schemas import RetrievalResult


class RAGService:
    """RAG 检索入口，组合 embedder 与 Milvus Collection。"""

    def __init__(self, embedder: LocalEmbedder, collection: Any) -> None:
        self._retriever = MultiRetriever(collection, embedder)

    def register_documents(self, id_to_text: dict[str, str]) -> None:
        """注册 BM25 关键词索引（id -> 文本）。"""
        self._retriever.register_keyword_documents(id_to_text)
        logger.info(
            "rag_service.py::register_documents 已注册 BM25 文档数={}",
            len(id_to_text),
        )

    async def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """按模式检索，返回检索结果列表。"""
        results = await self._retriever.retrieve(query, top_k=top_k, mode=mode)
        logger.info(
            "rag_service.py::retrieve 检索完成 mode={} top_k={} hits={}",
            mode,
            top_k,
            len(results),
        )
        return results
