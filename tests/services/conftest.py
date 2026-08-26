# -*- coding: utf-8 -*-
"""F3 · DocumentService 测试夹具：数据库、Milvus、embedding 与服务实例。"""

from uuid import uuid4

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.infrastructure.database.models import Base
from app.infrastructure.database.session import init_engine
from app.infrastructure.embedding import LocalEmbedder
from app.infrastructure.vectordb import MilvusManager
from app.services import DocumentService

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db"


@pytest_asyncio.fixture
async def db_session():
    """每个测试独立引擎与会话，并确保表存在。"""
    engine = init_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def milvus() -> MilvusManager:
    """连接本地 Milvus。"""
    manager = MilvusManager(host="localhost", port="19530")
    yield manager


@pytest_asyncio.fixture
async def collection(milvus: MilvusManager) -> str:
    """每个测试独立集合，结束后删除。"""
    name = f"test_f3_{uuid4().hex[:8]}"
    await milvus.create_collection(name, dim=512)
    yield name
    await milvus.drop_collection(name)


@pytest_asyncio.fixture
async def embedder() -> LocalEmbedder:
    """本地 embedding 模型，离线加载。"""
    return LocalEmbedder(
        model_name="BAAI/bge-small-zh-v1.5",
        device=None,
        expected_dim=512,
    )


@pytest_asyncio.fixture
async def service(
    embedder: LocalEmbedder,
    milvus: MilvusManager,
    collection: str,
) -> DocumentService:
    """F3 被测服务实例。"""
    return DocumentService(embedder, milvus, collection)
