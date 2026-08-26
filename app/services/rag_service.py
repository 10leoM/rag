# -*- coding: utf-8 -*-
"""RAG 检索服务：封装多路检索、重排与答案生成。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.rag.generator import RAGGenerator
from app.core.rag.reranker import Reranker
from app.core.rag.retriever import MultiRetriever
from app.infrastructure.embedding import LocalEmbedder
from app.models.schemas import RAGResponse, RetrievalResult


class OpenAILLMAdapter:
    """把 ModelRouter 适配为 RAGGenerator 需要的 ainvoke 接口。"""

    def __init__(self, router: Any, model: str | None = None) -> None:
        self._router = router
        self._model = model

    async def ainvoke(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        resp = await self._router.chat(
            messages,
            model_preference=self._model,
            temperature=kwargs.get("temperature", 0.3),
        )
        return resp.content


class RAGService:
    """RAG 检索入口，组合 embedder、Milvus、重排与生成。"""

    def __init__(
        self,
        embedder: LocalEmbedder,
        collection: Any,
        llm: Any | None = None,
        reranker_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self._retriever = MultiRetriever(collection, embedder)
        self._reranker = Reranker(model_name=reranker_model)
        self._generator = (
            RAGGenerator(llm, model_name="openai") if llm is not None else None
        )

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

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """对检索结果做 CrossEncoder 重排。"""
        reranked = await self._reranker.rerank(query, results, top_k=top_k)
        logger.info("rag_service.py::rerank 重排完成 hits={}", len(reranked))
        return reranked

    async def answer(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> RAGResponse:
        """检索 -> 重排 -> 生成，返回带引用的答案。"""
        if self._generator is None:
            raise RuntimeError("未注入 LLM，无法生成答案")
        # 1. 检索
        results = await self.retrieve(query, mode=mode, top_k=top_k)
        # 2. 重排
        reranked = await self.rerank(query, results, top_k=rerank_top_k)
        # 3. 生成
        response = await self._generator.generate(query, reranked, chat_history=[])
        logger.info(
            "rag_service.py::answer 生成完成 answer_len={} citations={}",
            len(response.answer),
            len(response.citations),
        )
        return response
