# -*- coding: utf-8 -*-
"""pytest 共享夹具：session 级加载一次模型，避免测试间重复下载与加载。"""

import json
from pathlib import Path

import pytest

from app.infrastructure.embedding import LocalEmbedder

DATA_DIR = Path(__file__).resolve().parent / "data" / "embedding"


@pytest.fixture(scope="session")
def embedder() -> LocalEmbedder:
    """加载一次本地 embedding 模型，供 F1 各测试复用。"""
    return LocalEmbedder(
        model_name="BAAI/bge-small-zh-v1.5",
        device=None,
        expected_dim=512,
    )


@pytest.fixture(scope="session")
def similarity_pairs() -> dict:
    """加载 F1 相似度测试数据。"""
    path = DATA_DIR / "similarity_pairs.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
