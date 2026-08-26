# -*- coding: utf-8 -*-
"""Milvus 测试夹具：连接、集合生命周期与测试记录。"""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest_asyncio

from app.infrastructure.vectordb import MilvusManager, MilvusRecord

DIM = 512
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "vectordb"


@pytest_asyncio.fixture
async def manager() -> MilvusManager:
    """连接本地 Milvus。"""
    m = MilvusManager(host="localhost", port="19530")
    yield m


@pytest_asyncio.fixture
async def collection(manager: MilvusManager) -> str:
    """每个测试独立集合，结束后删除。"""
    name = f"test_f2_{uuid4().hex[:8]}"
    await manager.create_collection(name, dim=DIM)
    yield name
    await manager.drop_collection(name)


@pytest_asyncio.fixture
async def chunk_record() -> MilvusRecord:
    """构造一条测试分块记录；向量用固定 seed 保证可复现。"""
    path = DATA_DIR / "chunk_record.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rng = np.random.default_rng(42)
    embedding = rng.random(DIM).astype(np.float32).tolist()
    return MilvusRecord(
        id=data["id"],
        embedding=embedding,
        text=data["text"],
        document_id=data["document_id"],
        chunk_index=data["chunk_index"],
        filename=data["filename"],
        mime_type=data["mime_type"],
        metadata=data["metadata"],
    )
