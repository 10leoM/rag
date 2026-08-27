# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化异步数据库引擎。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import select

from app.api.routes import chat, document, health, rag
from app.config import get_settings
from app.infrastructure.database.models import DocumentChunk
from app.infrastructure.database.session import configure_session, init_engine
from app.infrastructure.embedding import LocalEmbedder
from app.infrastructure.vectordb import MilvusManager
from app.services import OpenAILLMAdapter, RAGService

#应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    # 1. 数据库引擎
    engine = init_engine(settings.database_url)
    configure_session(engine)
    app.state.engine = engine
    # 2. embedding 与 Milvus 集合
    embedder = LocalEmbedder(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        expected_dim=settings.embedding_dim,
    )
    milvus = MilvusManager(host=settings.milvus_host, port=str(settings.milvus_port))
    await milvus.create_collection(settings.milvus_collection_name, settings.embedding_dim)
    collection = milvus.get_collection(settings.milvus_collection_name)
    # 3. 生成 LLM（未配置 Key 时仅支持检索）
    llm = None
    if settings.openai_api_key:
        from app.infrastructure.llm.model_router import ModelConfig, ModelRouter

        cfg = ModelConfig(
            model_id=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base or None,
            priority=0,
            weight=1.0,
        )
        llm = OpenAILLMAdapter(ModelRouter([cfg]), settings.openai_model)
    # 4. RAG 服务并注册 BM25 文档
    rag_service = RAGService(embedder, collection, llm=llm)
    from app.infrastructure.database.session import async_session_factory

    async with async_session_factory() as session:
        result = await session.execute(select(DocumentChunk.id, DocumentChunk.content))
        id_to_text = {str(row[0]): str(row[1]) for row in result.all()}
    rag_service.register_documents(id_to_text)
    app.state.rag_service = rag_service
    yield
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)

#应用创建
def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(rag.router, prefix=settings.api_prefix)
    return application


app = create_app()
