# -*- coding: utf-8 -*-
"""RAG 查询 API：检索、重排、生成的独立端点。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.models.schemas import RAGQueryRequest, RAGResponse
from app.services import RAGService

router = APIRouter(tags=["rag"])


def get_rag_service(request: Request) -> RAGService:
    """从应用状态读取初始化好的 RAG 服务。"""
    return request.app.state.rag_service


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    body: RAGQueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """执行 RAG 查询，返回答案、引用与可选中间态。"""
    return await service.answer(
        body.query,
        mode=body.mode,
        top_k=body.top_k,
        rerank_top_k=body.rerank_top_k,
        include_trace=body.include_trace,
    )
